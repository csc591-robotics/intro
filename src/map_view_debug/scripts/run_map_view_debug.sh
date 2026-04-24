#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────
# run_map_view_debug.sh — debug get_map_b64 by clicking in RViz.
#
# Brings up Gazebo + RViz aligned via the world_to_map sidecar (NO robot),
# then starts map_view_debug_node which renders the LLM map view from each
# "2D Pose Estimate" click and dumps the PNG + JSON to disk.
#
# Reuses nav2_llm_demo's parse_map_poses.py so the SOURCE_X/Y, DEST_X/Y,
# MAP_OFFSET_X/Y, MAP_YAML_PATH, WORLD_FILE and GAZEBO_MODEL_PATH used here
# are byte-identical to what the LLM agent runner uses.
#
# Usage:
#   bash src/map_view_debug/scripts/run_map_view_debug.sh <MAP_NAME>
#
# Examples:
#   bash src/map_view_debug/scripts/run_map_view_debug.sh warehouse
#
# Env overrides:
#   MAP_POSES_PATH        Override path to map_poses.yaml.
#   GAZEBO_MODEL_PATH     Manual override (auto-detected otherwise).
#   LAUNCH_RVIZ=false     Skip RViz (rarely useful here).
#   LAUNCH_GAZEBO=false   Skip Gazebo (RViz + debug node only).
#   USE_SIM_TIME=false    Use wall clock instead of /clock.
#   CROP_RADIUS_M=18.0    Override render crop radius.
#   OUTPUT_SIZE=512       Override rendered PNG side length.
#   OUTPUT_DIR=...        Override where pose_NNN.png/.json land.
# ──────────────────────────────────────────────────────────────────────────

set -eo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <MAP_NAME>" >&2
  exit 1
fi

MAP_NAME_IN="$1"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PARSER="${WORKSPACE_ROOT}/src/nav2_llm_demo/scripts/parse_map_poses.py"

if [[ ! -f "$PARSER" ]]; then
  echo "parse_map_poses.py not found at $PARSER" >&2
  echo "(map_view_debug reuses nav2_llm_demo's parser; both packages must be in src/.)" >&2
  exit 1
fi

if [[ -n "${ROS_DISTRO:-}" && -f "/opt/ros/${ROS_DISTRO}/setup.bash" ]]; then
  # shellcheck disable=SC1091
  source "/opt/ros/${ROS_DISTRO}/setup.bash"
fi

if [[ -f "${WORKSPACE_ROOT}/install/setup.bash" ]]; then
  # shellcheck disable=SC1091
  source "${WORKSPACE_ROOT}/install/setup.bash"
fi

if ! command -v ros2 >/dev/null 2>&1; then
  echo "ERROR: 'ros2' not on PATH. Source /opt/ros/<distro>/setup.bash first." >&2
  exit 1
fi

PKG_PREFIX="$(ros2 pkg prefix map_view_debug 2>/dev/null || true)"
if [[ -z "$PKG_PREFIX" ]]; then
  echo "ERROR: map_view_debug not found by ros2." >&2
  echo "       Build it and source the matching install/setup.bash:" >&2
  echo "         colcon build --packages-select map_view_debug nav2_llm_demo world_to_map" >&2
  echo "         source install/setup.bash" >&2
  exit 1
fi
DEBUG_NODE_EXE="${PKG_PREFIX}/lib/map_view_debug/map_view_debug_node"
if [[ ! -x "$DEBUG_NODE_EXE" ]]; then
  echo "ERROR: map_view_debug_node executable not found at $DEBUG_NODE_EXE" >&2
  echo "       Rebuild: colcon build --packages-select map_view_debug" >&2
  exit 1
fi

if ! python3 -c "from nav2_llm_demo.llm.map_renderer import render_annotated_map" 2>/tmp/mvd_import_err; then
  echo "ERROR: nav2_llm_demo.llm.map_renderer failed to import:" >&2
  cat /tmp/mvd_import_err >&2
  echo >&2
  echo "Run: colcon build --packages-select nav2_llm_demo" >&2
  exit 1
fi

PARSER_ARGS=("$MAP_NAME_IN")
if [[ -n "${MAP_POSES_PATH:-}" ]]; then
  PARSER_ARGS+=("$MAP_POSES_PATH")
fi

PARSER_OUT="$(python3 "$PARSER" "${PARSER_ARGS[@]}")"
eval "$PARSER_OUT"

if [[ -z "${MAP_YAML_PATH}" || ! -f "${MAP_YAML_PATH}" ]]; then
  echo "Could not resolve map YAML for ${MAP_NAME_IN} (got '${MAP_YAML_PATH}')." >&2
  exit 1
fi

LAUNCH_GAZEBO="${LAUNCH_GAZEBO:-true}"
if [[ "${HAS_WORLD}" != "1" && "${LAUNCH_GAZEBO}" == "true" ]]; then
  echo "WARNING: ${MAP_NAME_IN} has no Gazebo world on disk (sidecar references" >&2
  echo "         ${WORLD_FILE:-<none>}). Falling back to LAUNCH_GAZEBO=false." >&2
  LAUNCH_GAZEBO="false"
fi

if [[ -z "${GAZEBO_MODEL_PATH:-}" && -n "${GAZEBO_MODEL_PATH_AUTO}" ]]; then
  GAZEBO_MODEL_PATH="${GAZEBO_MODEL_PATH_AUTO}"
fi

LAUNCH_RVIZ="${LAUNCH_RVIZ:-true}"
USE_SIM_TIME="${USE_SIM_TIME:-true}"
CROP_RADIUS_M="${CROP_RADIUS_M:-18.0}"
OUTPUT_SIZE="${OUTPUT_SIZE:-512}"
OUTPUT_DIR="${OUTPUT_DIR:-}"

echo "──────────────────────────────────────────────────────────────────────"
echo "  map_view_debug — render LLM map view from RViz clicks"
echo "──────────────────────────────────────────────────────────────────────"
echo "  Map name              : ${MAP_NAME}"
echo "  map_poses.yaml        : ${MAP_POSES_PATH}"
echo "  Map YAML              : ${MAP_YAML_PATH}"
if [[ -n "${SIDECAR_PATH}" ]]; then
  echo "  Sidecar               : ${SIDECAR_PATH}"
  echo "  world_to_map_offset   : (${MAP_OFFSET_X}, ${MAP_OFFSET_Y})"
fi
if [[ "${LAUNCH_GAZEBO}" == "true" ]]; then
  echo "  Gazebo world          : ${WORLD_FILE}"
else
  echo "  Gazebo                : (skipped)"
fi
echo "  Source (map frame)    : (${SOURCE_X}, ${SOURCE_Y})"
echo "  Destination (map)     : (${DEST_X}, ${DEST_Y})"
echo "  static map->odom TF   : (${STATIC_TF_X}, ${STATIC_TF_Y})"
if [[ -n "${GAZEBO_MODEL_PATH:-}" ]]; then
  echo "  GAZEBO_MODEL_PATH     : ${GAZEBO_MODEL_PATH}"
fi
echo "  Launch RViz           : ${LAUNCH_RVIZ}"
echo "  crop_radius_m         : ${CROP_RADIUS_M}"
echo "  output_size           : ${OUTPUT_SIZE}"
if [[ -n "${OUTPUT_DIR}" ]]; then
  echo "  output_dir            : ${OUTPUT_DIR}"
else
  WS_DIR_DEFAULT="${WORKSPACE_DIR:-/workspace}"
  echo "  output_dir            : (auto: ${WS_DIR_DEFAULT}/map_view_debug_runs/<timestamp>/)"
fi
echo "──────────────────────────────────────────────────────────────────────"
echo "  Click the '2D Pose Estimate' tool in RViz (shortcut: E) and drag"
echo "  an arrow on the map. Each click writes pose_NNN.png + .json."
echo "──────────────────────────────────────────────────────────────────────"

LAUNCH_ARGS=(
  "map_yaml:=${MAP_YAML_PATH}"
  "source_x:=${SOURCE_X}"
  "source_y:=${SOURCE_Y}"
  "dest_x:=${DEST_X}"
  "dest_y:=${DEST_Y}"
  "map_offset_x:=${STATIC_TF_X}"
  "map_offset_y:=${STATIC_TF_Y}"
  "use_sim_time:=${USE_SIM_TIME}"
  "launch_rviz:=${LAUNCH_RVIZ}"
  "launch_gazebo:=${LAUNCH_GAZEBO}"
  "crop_radius_m:=${CROP_RADIUS_M}"
  "output_size:=${OUTPUT_SIZE}"
)

if [[ "${LAUNCH_GAZEBO}" == "true" ]]; then
  LAUNCH_ARGS+=("world:=${WORLD_FILE}")
fi

if [[ -n "${GAZEBO_MODEL_PATH:-}" ]]; then
  LAUNCH_ARGS+=("gazebo_model_path:=${GAZEBO_MODEL_PATH}")
fi

if [[ -n "${OUTPUT_DIR}" ]]; then
  LAUNCH_ARGS+=("output_dir:=${OUTPUT_DIR}")
fi

exec ros2 launch map_view_debug map_view_debug.launch.py "${LAUNCH_ARGS[@]}"
