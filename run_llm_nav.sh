#!/usr/bin/env bash
set -euo pipefail

# Resolve the repo/workspace directory so the script works from any current path.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="${WORKSPACE_DIR:-$SCRIPT_DIR}"

# Allow a mission string to be passed as the first argument.
# If nothing is passed, use the default demo request.
MISSION_REQUEST="${1:-Reach the far side of the obstacle course}"

# Keep the ROS launch options configurable without editing the script.
USE_SIM_TIME="${USE_SIM_TIME:-True}"
USE_RVIZ="${USE_RVIZ:-False}"
DEFAULT_NAV2_MAP="${WORKSPACE_DIR}/src/nav2_llm_demo/config/custom_map.yaml"
NAV2_MAP="${NAV2_MAP:-$DEFAULT_NAV2_MAP}"
NAV2_PARAMS_FILE="${NAV2_PARAMS_FILE:-}"
INITIAL_POSE_X="${INITIAL_POSE_X:-0.0}"
INITIAL_POSE_Y="${INITIAL_POSE_Y:-0.0}"
INITIAL_POSE_Z="${INITIAL_POSE_Z:-0.0}"
INITIAL_POSE_QZ="${INITIAL_POSE_QZ:-0.0}"
INITIAL_POSE_QW="${INITIAL_POSE_QW:-1.0}"
INITIAL_POSE_RETRIES="${INITIAL_POSE_RETRIES:-5}"

# These sleeps are simple startup buffers so later processes do not race
# earlier ones while Gazebo/Nav2 are still initializing.
NAV2_STARTUP_WAIT_SEC="${NAV2_STARTUP_WAIT_SEC:-8}"
LLM_NODE_WAIT_SEC="${LLM_NODE_WAIT_SEC:-5}"

cleanup() {
  local exit_code=$?
  # Tear down background ROS processes when the script exits or is interrupted.
  # This keeps one Ctrl+C from leaving Gazebo/Nav2/topic echo running.
  if [[ -n "${STATUS_PID:-}" ]]; then
    kill "$STATUS_PID" >/dev/null 2>&1 || true
  fi
  if [[ -n "${LLM_NAV_PID:-}" ]]; then
    kill "$LLM_NAV_PID" >/dev/null 2>&1 || true
  fi
  if [[ -n "${NAV2_PID:-}" ]]; then
    kill "$NAV2_PID" >/dev/null 2>&1 || true
  fi
  if [[ -n "${GAZEBO_PID:-}" ]]; then
    kill "$GAZEBO_PID" >/dev/null 2>&1 || true
  fi
  wait >/dev/null 2>&1 || true
  exit "$exit_code"
}

trap cleanup EXIT INT TERM

INITIAL_POSE_PAYLOAD=$(cat <<EOF
{header: {frame_id: map}, pose: {pose: {position: {x: $INITIAL_POSE_X, y: $INITIAL_POSE_Y, z: $INITIAL_POSE_Z}, orientation: {x: 0.0, y: 0.0, z: $INITIAL_POSE_QZ, w: $INITIAL_POSE_QW}}, covariance: [0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1]}}
EOF
)

# Enter the mounted ROS workspace and load the base ROS environment.
cd "$WORKSPACE_DIR"

# ROS setup scripts are not safe under `set -u` in this environment,
# so temporarily disable nounset while sourcing them.
set +u
source /opt/ros/humble/setup.bash
set -u

# The package must already be built so ROS can find the installed nodes.
if [[ ! -f install/setup.bash ]]; then
  echo "Missing install/setup.bash. Build the workspace before running this script."
  exit 1
fi

# Load the workspace overlay so `ros2 run` can see this repo's packages.
set +u
source install/setup.bash
set -u

# Export variables from `.env` into this shell before launching ROS processes.
# That is how `GROQ_API_KEY` reaches the Python node at runtime.
if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi

# The LLM node reads the key from the environment if it is not given as a ROS param.
if [[ -z "${GROQ_API_KEY:-}" ]]; then
  echo "GROQ_API_KEY is not set. Add it to ${WORKSPACE_DIR}/.env before launching."
  exit 1
fi

# Resolve the config files that the `llm_nav_node` needs at startup.
PARAMS_FILE="${WORKSPACE_DIR}/src/nav2_llm_demo/config/llm_nav_params.yaml"
ROUTE_GRAPH_FILE="${WORKSPACE_DIR}/src/nav2_llm_demo/config/route_graph.json"

if [[ -z "$NAV2_MAP" ]]; then
  echo "NAV2_MAP is not set."
  echo "Set NAV2_MAP to the map yaml Nav2 should use before launching."
  exit 1
fi

if [[ ! -f "$NAV2_MAP" ]]; then
  echo "Nav2 map file not found: $NAV2_MAP"
  exit 1
fi

if [[ -n "$NAV2_PARAMS_FILE" && ! -f "$NAV2_PARAMS_FILE" ]]; then
  echo "Nav2 params file not found: $NAV2_PARAMS_FILE"
  exit 1
fi

if [[ ! -f "$PARAMS_FILE" ]]; then
  echo "Missing params file: $PARAMS_FILE"
  exit 1
fi

if [[ ! -f "$ROUTE_GRAPH_FILE" ]]; then
  echo "Missing route graph file: $ROUTE_GRAPH_FILE"
  exit 1
fi

# Step 1: start the simulator world and robot model.
# This is the long-running Gazebo process.
echo "Starting Gazebo + TurtleBot3..."
ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py &
GAZEBO_PID=$!

# Give Gazebo a moment to come up before bringing in navigation.
sleep 5

# Step 2: start Nav2 and AMCL.
# This handles localization and path execution; the LLM node does not replace it.
echo "Starting Nav2 + AMCL..."
nav2_launch_args=(
  use_sim_time:="$USE_SIM_TIME"
  use_rviz:="$USE_RVIZ"
  map:="$NAV2_MAP"
)

if [[ -n "$NAV2_PARAMS_FILE" ]]; then
  nav2_launch_args+=(params_file:="$NAV2_PARAMS_FILE")
fi

ros2 launch turtlebot3_navigation2 navigation2.launch.py \
  "${nav2_launch_args[@]}" &
NAV2_PID=$!

# Allow Nav2 lifecycle nodes time to initialize before starting the high-level controller.
sleep "$NAV2_STARTUP_WAIT_SEC"

# Seed AMCL with the robot's initial pose on the saved map so the `map`
# transform becomes available without a separate manual publish step.
echo "Publishing initial pose for AMCL..."
for ((i = 1; i <= INITIAL_POSE_RETRIES; i++)); do
  echo "Initial pose attempt ${i}/${INITIAL_POSE_RETRIES}..."
  ros2 topic pub --once /initialpose geometry_msgs/msg/PoseWithCovarianceStamped \
    "$INITIAL_POSE_PAYLOAD"
  sleep 1
done

# Step 3: start the Groq-backed decision node.
# This node listens for `/navigation_request`, asks the LLM for a legal route,
# and sends checkpoint goals to Nav2 one segment at a time.
echo "Starting Groq decision node..."
ros2 run nav2_llm_demo llm_nav_node \
  --ros-args \
  --params-file "$PARAMS_FILE" \
  -p route_graph_path:="$ROUTE_GRAPH_FILE" &
LLM_NAV_PID=$!

# Give the node time to subscribe and wait for Nav2 to report active.
sleep "$LLM_NODE_WAIT_SEC"

# Optional visibility: stream status updates in the same terminal session.
echo "Streaming /navigation_status in the background..."
ros2 topic echo /navigation_status &
STATUS_PID=$!

# Step 4: publish a one-shot mission request so the node immediately has work to do.
if [[ -n "$MISSION_REQUEST" ]]; then
  echo "Publishing mission request: $MISSION_REQUEST"
  ros2 topic pub --once /navigation_request std_msgs/msg/String \
    "{data: '$MISSION_REQUEST'}"
fi

# Keep the shell attached to the background jobs until the user stops the script.
echo "System is running. Press Ctrl+C to stop Gazebo, Nav2, llm_nav_node, and status echo."
wait
