#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# nav2_llm_demo runner — brings up Gazebo + RViz + Nav2 + the LLM controller
# all aligned to the source pose recorded in map_poses.yaml so the robot
# starts at the **same place in both Gazebo and RViz** from frame zero.
#
# Usage (inside the Docker container, after building):
#
#   cd /workspace/intro
#   source install/setup.bash
#
#   # Most common case — a world_to_map-generated map:
#   bash src/nav2_llm_demo/scripts/run_llm_nav.sh warehouse
#
#   # A specific PGM filename also works:
#   bash src/nav2_llm_demo/scripts/run_llm_nav.sh warehouse.pgm
#
#   # Hand-crafted map (no Gazebo) needs the Nav2 map yaml as a fallback:
#   MAP_YAML_FALLBACK=src/custom_map_builder/maps/diamond_blocked.yaml \
#     bash src/nav2_llm_demo/scripts/run_llm_nav.sh diamond_blocked
#
# The runner reads the entry for <map_name> in
#   src/custom_map_builder/maps/map_poses.yaml
# and forwards everything to llm_nav.launch.py.  That launch file then:
#   - launches Gazebo with the world file referenced in the sidecar,
#   - spawns the TurtleBot3 at the recorded source pose,
#   - publishes a static map -> odom TF aligned with the spawn,
#   - opens RViz on the rasterized PGM,
#   - starts Nav2 navigation servers (no AMCL — Gazebo is ground truth),
#   - starts llm_nav_node parameterised with the same map_poses.yaml entry.
#
# Optional environment overrides:
#   MAP_POSES_PATH      Path to map_poses.yaml (default: shared file).
#   NAV2_PARAMS_FILE    Path to Nav2 params yaml.
#   LLM_PARAMS_FILE     Path to llm_nav_params.yaml (default: package share).
#   MAP_YAML_FALLBACK   Required for hand-crafted maps with sidecar: null.
#   GAZEBO_MODEL_PATH   Extra model dirs (auto-detected if unset).
#   LAUNCH_RVIZ         true (default) | false.
#   USE_SIM_TIME        true (default) | false.
# ---------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"

if [[ $# -lt 1 ]]; then
  echo "Usage: bash $0 <map_name>"
  echo ""
  echo "Examples:"
  echo "  bash $0 warehouse"
  echo "  bash $0 workshop_example"
  echo "  bash $0 test_zone"
  exit 1
fi

MAP_NAME="$1"

MAP_POSES_PATH="${MAP_POSES_PATH:-$WORKSPACE_DIR/src/custom_map_builder/maps/map_poses.yaml}"
NAV2_PARAMS_FILE="${NAV2_PARAMS_FILE:-/opt/ros/humble/share/nav2_bringup/params/nav2_params.yaml}"
LLM_PARAMS_FILE="${LLM_PARAMS_FILE:-}"
MAP_YAML_FALLBACK="${MAP_YAML_FALLBACK:-}"
LAUNCH_RVIZ="${LAUNCH_RVIZ:-true}"
USE_SIM_TIME="${USE_SIM_TIME:-true}"

# Auto-detect the gazebo_models_worlds_collection so meshed worlds resolve
# their model:// URIs without the user having to export anything.
if [[ -z "${GAZEBO_MODEL_PATH:-}" ]]; then
  for CANDIDATE in \
    "$WORKSPACE_DIR/world_files/gazebo_models_worlds_collection/models" \
    "$WORKSPACE_DIR/intro/world_files/gazebo_models_worlds_collection/models"; do
    if [[ -d "$CANDIDATE" ]]; then
      GAZEBO_MODEL_PATH="$CANDIDATE"
      break
    fi
  done
fi
GAZEBO_MODEL_PATH="${GAZEBO_MODEL_PATH:-}"

if [[ ! -f "$MAP_POSES_PATH" ]]; then
  echo "ERROR: map_poses.yaml not found at $MAP_POSES_PATH"
  echo "       Override with MAP_POSES_PATH=/path/to/map_poses.yaml"
  exit 1
fi

if [[ ! -f "$NAV2_PARAMS_FILE" ]]; then
  echo "ERROR: Nav2 params file not found at $NAV2_PARAMS_FILE"
  echo "       Override with NAV2_PARAMS_FILE=/path/to/nav2_params.yaml"
  exit 1
fi

# ── Source ROS ────────────────────────────────────────────────────────────
cd "$WORKSPACE_DIR"
set +u
source /opt/ros/humble/setup.bash
if [[ -f install/setup.bash ]]; then
  source install/setup.bash
else
  echo "WARNING: install/setup.bash not found. Build the workspace first:"
  echo "  colcon build --packages-select nav2_llm_demo world_to_map"
fi
set -u

export TURTLEBOT3_MODEL="${TURTLEBOT3_MODEL:-burger}"

echo "================================================================"
echo "nav2_llm_demo runner"
echo "  map_name           : $MAP_NAME"
echo "  map_poses_path     : $MAP_POSES_PATH"
echo "  nav2_params_file   : $NAV2_PARAMS_FILE"
echo "  use_sim_time       : $USE_SIM_TIME"
echo "  launch_rviz        : $LAUNCH_RVIZ"
echo "  GAZEBO_MODEL_PATH  : ${GAZEBO_MODEL_PATH:-<unset>}"
[[ -n "$MAP_YAML_FALLBACK" ]] && \
  echo "  map_yaml_fallback  : $MAP_YAML_FALLBACK"
[[ -n "$LLM_PARAMS_FILE" ]] && \
  echo "  llm_params_file    : $LLM_PARAMS_FILE"
echo "================================================================"
echo ""
echo "  - Gazebo and RViz will both open."
echo "  - The TurtleBot3 will spawn at the 'source' pose recorded in"
echo "    map_poses.yaml (Gazebo world frame), and the map -> odom TF will"
echo "    place it at the same spot on the rasterized PGM in RViz."
echo "  - Send a mission with:"
echo "      ros2 topic pub --once /navigation_request std_msgs/String \\"
echo "        \"data: '<your goal alias>'\""
echo ""

LAUNCH_ARGS=(
  "map_poses_path:=$MAP_POSES_PATH"
  "map_name:=$MAP_NAME"
  "nav2_params_file:=$NAV2_PARAMS_FILE"
  "use_sim_time:=$USE_SIM_TIME"
  "launch_rviz:=$LAUNCH_RVIZ"
)
[[ -n "$LLM_PARAMS_FILE" ]] && LAUNCH_ARGS+=("llm_params_file:=$LLM_PARAMS_FILE")
[[ -n "$MAP_YAML_FALLBACK" ]] && LAUNCH_ARGS+=("map_yaml_fallback:=$MAP_YAML_FALLBACK")
[[ -n "$GAZEBO_MODEL_PATH" ]] && LAUNCH_ARGS+=("gazebo_model_path:=$GAZEBO_MODEL_PATH")

ros2 launch nav2_llm_demo llm_nav.launch.py "${LAUNCH_ARGS[@]}"
