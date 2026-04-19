#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# LLM Agent Navigation Runner
#
# Usage:
#   bash ./run_llm_nav.sh MAP_NAME
#   bash ./run_llm_nav.sh diamond_blocked
#
# MAP_NAME is looked up in src/nav2_llm_demo/maps/nav_config.yaml.
# The script starts Gazebo, map_server, and the LLM vision agent.
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="${WORKSPACE_DIR:-$SCRIPT_DIR}"

# The map name is the first argument (required).
MAP_NAME="${1:-}"
if [[ -z "$MAP_NAME" ]]; then
  echo "Usage: bash ./run_llm_nav.sh MAP_NAME"
  echo ""
  echo "MAP_NAME must match an entry in src/nav2_llm_demo/maps/nav_config.yaml."
  echo "Example: bash ./run_llm_nav.sh diamond_blocked"
  exit 1
fi

NAV_CONFIG="${WORKSPACE_DIR}/src/nav2_llm_demo/maps/nav_config.yaml"
PARSE_SCRIPT="${WORKSPACE_DIR}/src/nav2_llm_demo/scripts/parse_nav_config.py"

USE_SIM_TIME="${USE_SIM_TIME:-True}"
USE_RVIZ="${USE_RVIZ:-True}"
GAZEBO_STARTUP_WAIT="${GAZEBO_STARTUP_WAIT:-5}"

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

cleanup() {
  local exit_code=$?
  if [[ -n "${STATUS_PID:-}" ]]; then
    kill "$STATUS_PID" >/dev/null 2>&1 || true
  fi
  if [[ -n "${AGENT_PID:-}" ]]; then
    kill "$AGENT_PID" >/dev/null 2>&1 || true
  fi
  if [[ -n "${GAZEBO_PID:-}" ]]; then
    kill "$GAZEBO_PID" >/dev/null 2>&1 || true
  fi
  wait >/dev/null 2>&1 || true
  exit "$exit_code"
}

trap cleanup EXIT INT TERM

# ---------------------------------------------------------------------------
# Source ROS + workspace
# ---------------------------------------------------------------------------

cd "$WORKSPACE_DIR"

set +u
source /opt/ros/humble/setup.bash
set -u

if [[ ! -f install/setup.bash ]]; then
  echo "Missing install/setup.bash. Build the workspace first:"
  echo "  colcon build --packages-select nav2_llm_demo"
  exit 1
fi

set +u
source install/setup.bash
set -u

if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi

export TURTLEBOT3_MODEL="${TURTLEBOT3_MODEL:-burger}"

# ---------------------------------------------------------------------------
# Validate LLM env
# ---------------------------------------------------------------------------

if [[ -z "${LLM_PROVIDER:-}" ]]; then
  echo "LLM_PROVIDER is not set. Add it to ${WORKSPACE_DIR}/.env"
  exit 1
fi

if [[ -z "${LLM_MODEL:-}" ]]; then
  echo "LLM_MODEL is not set. Add it to ${WORKSPACE_DIR}/.env"
  exit 1
fi

# ---------------------------------------------------------------------------
# Parse map config
# ---------------------------------------------------------------------------

if [[ ! -f "$NAV_CONFIG" ]]; then
  echo "Map config not found: $NAV_CONFIG"
  exit 1
fi

if [[ ! -f "$PARSE_SCRIPT" ]]; then
  echo "Config parser not found: $PARSE_SCRIPT"
  exit 1
fi

echo "Loading config for map: $MAP_NAME"
eval "$(python3 "$PARSE_SCRIPT" "$MAP_NAME" "$NAV_CONFIG")"

if [[ -z "${MAP_YAML_PATH:-}" ]]; then
  echo "Failed to parse config for map '$MAP_NAME'."
  exit 1
fi

if [[ ! -f "$MAP_YAML_PATH" ]]; then
  echo "Map YAML not found: $MAP_YAML_PATH"
  echo "Copy the map files from custom_map_builder/maps/ into src/nav2_llm_demo/maps/."
  exit 1
fi

echo "  Map YAML : $MAP_YAML_PATH"
echo "  Source   : ($SOURCE_X, $SOURCE_Y) yaw=$SOURCE_YAW"
echo "  Dest     : ($DEST_X, $DEST_Y) yaw=$DEST_YAW"

# ---------------------------------------------------------------------------
# Step 1: Gazebo (empty world + TurtleBot3 spawned at source pose)
# ---------------------------------------------------------------------------

export GAZEBO_MODEL_PATH=$GAZEBO_MODEL_PATH:/workspace/gazebo_models_worlds_collection/models

echo ""
echo "Starting Gazebo (empty world) + TurtleBot3 at source pose..."
# launch hardcoded launchfile to warehouse world
ros2 launch warehouse_world.launch.py \
  x_pose:="$SOURCE_X" \
  y_pose:="$SOURCE_Y" \
  &
GAZEBO_PID=$!

sleep "$GAZEBO_STARTUP_WAIT"

# ---------------------------------------------------------------------------
# Step 2: Launch map_server + agent node
# ---------------------------------------------------------------------------

echo "Starting map_server + LLM agent node (${LLM_PROVIDER}/${LLM_MODEL})..."
ros2 launch nav2_llm_demo llm_agent.launch.py \
  map_yaml:="$MAP_YAML_PATH" \
  source_x:="$SOURCE_X" \
  source_y:="$SOURCE_Y" \
  source_yaw:="$SOURCE_YAW" \
  dest_x:="$DEST_X" \
  dest_y:="$DEST_Y" \
  dest_yaw:="$DEST_YAW" \
  use_sim_time:="$USE_SIM_TIME" \
  launch_rviz:="$USE_RVIZ" \
  &
AGENT_PID=$!

sleep 3

# ---------------------------------------------------------------------------
# Step 3: Stream status
# ---------------------------------------------------------------------------

echo ""
echo "Streaming /navigation_status in the background..."
ros2 topic echo /navigation_status &
STATUS_PID=$!

# ---------------------------------------------------------------------------
# Keep alive
# ---------------------------------------------------------------------------

echo ""
echo "System is running.  Map: $MAP_NAME"
echo "Press Ctrl+C to stop everything."
wait
