#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────
# run_llm_nav.sh — launch the vision-based LLM agent.
#
# Reads custom_map_builder/maps/map_poses.yaml, resolves the per-world sidecar
# (when present), and brings up:
#
#   - Gazebo (the .world file referenced by the sidecar) + TurtleBot3 spawn
#   - nav2_map_server (PGM/YAML loaded into the map topic)
#   - static map -> odom TF aligned with the chosen source pose
#   - RViz with the map view
#   - llm_agent_node (vision LLM controlling /cmd_vel)
#
# For hand-crafted maps (sidecar==null) the script falls back to the legacy
# behaviour: map_server + identity TF + agent.  Gazebo is NOT started in that
# case (no .world is associated with the map).
#
# Usage:
#   bash src/nav2_llm_demo/scripts/run_llm_nav.sh <MAP_NAME> [--flow 1|2|3|4|5]
#
# Examples:
#   bash src/nav2_llm_demo/scripts/run_llm_nav.sh warehouse              # flow 1 (default)
#   bash src/nav2_llm_demo/scripts/run_llm_nav.sh warehouse --flow 2     # ReAct (image-in-tool)
#   bash src/nav2_llm_demo/scripts/run_llm_nav.sh warehouse --flow 3     # ReAct + LiDAR
#   bash src/nav2_llm_demo/scripts/run_llm_nav.sh warehouse --flow 4     # fixed gather/decide cycle
#   bash src/nav2_llm_demo/scripts/run_llm_nav.sh warehouse --flow 5     # A* + LLM follower
#   LLM_FLOW=5 bash src/nav2_llm_demo/scripts/run_llm_nav.sh warehouse   # via env
#   bash src/nav2_llm_demo/scripts/run_llm_nav.sh diamond_blocked
#
# Env overrides:
#   MAP_POSES_PATH        Override path to map_poses.yaml.
#   GAZEBO_MODEL_PATH     Manual override (auto-detected otherwise).
#   LAUNCH_RVIZ=false     Skip RViz.
#   USE_SIM_TIME=false    Use wall clock instead of /clock.
#   TURTLEBOT3_MODEL      burger (default) | waffle | waffle_pi
#   LLM_FLOW=1|2|3|4|5    Which agent flow to use; --flow CLI arg overrides this.
# ──────────────────────────────────────────────────────────────────────────

set -eo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <MAP_NAME> [--flow 1|2|3|4|5]" >&2
  exit 1
fi

MAP_NAME_IN="$1"
shift

while [[ $# -gt 0 ]]; do
  case "$1" in
    --flow)
      LLM_FLOW="$2"
      shift 2
      ;;
    --flow=*)
      LLM_FLOW="${1#*=}"
      shift
      ;;
    -h|--help)
      echo "Usage: $0 <MAP_NAME> [--flow 1|2|3|4|5]" >&2
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      echo "Usage: $0 <MAP_NAME> [--flow 1|2|3|4|5]" >&2
      exit 1
      ;;
  esac
done

LLM_FLOW="${LLM_FLOW:-1}"
case "${LLM_FLOW}" in
  1|2|3|4|5) ;;
  *)
    echo "ERROR: --flow must be 1, 2, 3, 4, or 5 (got '${LLM_FLOW}')." >&2
    exit 1
    ;;
esac
export LLM_FLOW

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PARSER="${SCRIPT_DIR}/parse_map_poses.py"

if [[ ! -f "$PARSER" ]]; then
  echo "parse_map_poses.py not found at $PARSER" >&2
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

# Source .env (LLM_PROVIDER, LLM_MODEL, *_API_KEY, etc.) so the values are
# inherited by every process the launch system spawns. Without this the
# llm_agent_node thread crashes with "LLM_PROVIDER not set".
ENV_FILE=""
for cand in "${WORKSPACE_ROOT}/.env" "${PWD}/.env"; do
  if [[ -f "$cand" ]]; then
    ENV_FILE="$cand"
    break
  fi
done
if [[ -n "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

if [[ -z "${LLM_PROVIDER:-}" || -z "${LLM_MODEL:-}" ]]; then
  echo "ERROR: LLM_PROVIDER and LLM_MODEL must be set." >&2
  echo "       Add them to ${WORKSPACE_ROOT}/.env or export them before running." >&2
  echo "       (See ${WORKSPACE_ROOT}/.env.example for an example.)" >&2
  exit 1
fi

# Pre-flight: confirm the llm_agent_node executable was actually installed by
# colcon. If not, we'd otherwise see Gazebo come up while the agent silently
# never starts.
LLM_AGENT_EXE="${WORKSPACE_ROOT}/install/nav2_llm_demo/lib/nav2_llm_demo/llm_agent_node"
if [[ ! -x "$LLM_AGENT_EXE" ]]; then
  echo "ERROR: llm_agent_node executable not found at $LLM_AGENT_EXE" >&2
  echo "       Run: colcon build --packages-select nav2_llm_demo" >&2
  exit 1
fi

# Pre-flight: confirm the dispatcher (which routes to the right flow_N
# package based on $LLM_FLOW) actually imports. The agent node otherwise
# crashes at module-import time with no obvious error reaching the launch
# screen.
#
# We use the per-flow effective provider for the helpful "pip install"
# hint so the suggestion matches what the chosen flow needs (e.g. flow_5
# uses anthropic via FLOW5_LLM_PROVIDER even if global LLM_PROVIDER is
# openai).
FLOW_PROVIDER_VAR="FLOW${LLM_FLOW}_LLM_PROVIDER"
EFFECTIVE_PROVIDER="${!FLOW_PROVIDER_VAR:-${LLM_PROVIDER}}"
if ! python3 -c "from nav2_llm_demo.llm import build_agent" 2>/tmp/llm_import_err; then
  echo "ERROR: nav2_llm_demo.llm failed to import (LLM_FLOW=${LLM_FLOW}):" >&2
  cat /tmp/llm_import_err >&2
  echo >&2
  echo "If the error mentions an unsupported LLM_FLOW, the installed" >&2
  echo "package is stale. Rebuild the workspace and re-source:" >&2
  echo "  colcon build --packages-select nav2_llm_demo" >&2
  echo "  source install/setup.bash" >&2
  echo >&2
  echo "If a langchain/provider import is missing, run:" >&2
  echo "  pip install langchain langchain-core langgraph Pillow numpy" >&2
  echo "  pip install langchain-${EFFECTIVE_PROVIDER}" >&2
  exit 1
fi

export TURTLEBOT3_MODEL="${TURTLEBOT3_MODEL:-burger}"

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

if [[ -z "${GAZEBO_MODEL_PATH:-}" && -n "${GAZEBO_MODEL_PATH_AUTO}" ]]; then
  GAZEBO_MODEL_PATH="${GAZEBO_MODEL_PATH_AUTO}"
fi

LAUNCH_RVIZ="${LAUNCH_RVIZ:-true}"
USE_SIM_TIME="${USE_SIM_TIME:-true}"

echo "──────────────────────────────────────────────────────────────────────"
echo "  nav2_llm_demo — LLM agent runner"
echo "──────────────────────────────────────────────────────────────────────"
echo "  Map name              : ${MAP_NAME}"
echo "  map_poses.yaml        : ${MAP_POSES_PATH}"
echo "  Map YAML              : ${MAP_YAML_PATH}"
if [[ -n "${SIDECAR_PATH}" ]]; then
  echo "  Sidecar               : ${SIDECAR_PATH}"
  echo "  world_to_map_offset   : (${MAP_OFFSET_X}, ${MAP_OFFSET_Y})"
  if [[ "${HAS_WORLD}" == "1" ]]; then
    echo "  Gazebo world          : ${WORLD_FILE}"
  else
    echo "  Gazebo world          : (sidecar references ${WORLD_FILE} — not "
    echo "                          found on disk; running legacy flow)"
  fi
else
  echo "  Sidecar               : (none — hand-crafted map, legacy flow)"
fi
echo "  Spawn (Gazebo world)  : (${SPAWN_X}, ${SPAWN_Y}, yaw=${SPAWN_YAW})"
echo "  Source (map frame)    : (${SOURCE_X}, ${SOURCE_Y}, yaw=${SOURCE_YAW})"
echo "  Destination (map)     : (${DEST_X}, ${DEST_Y}, yaw=${DEST_YAW})"
echo "  static map->odom TF   : (${STATIC_TF_X}, ${STATIC_TF_Y})"
if [[ -n "${GAZEBO_MODEL_PATH:-}" ]]; then
  echo "  GAZEBO_MODEL_PATH     : ${GAZEBO_MODEL_PATH}"
fi
echo "  TURTLEBOT3_MODEL      : ${TURTLEBOT3_MODEL}"
echo "  Launch RViz           : ${LAUNCH_RVIZ}"
echo "  LLM flow              : ${LLM_FLOW}  (logs -> intro/llm_agent_runs/flow_${LLM_FLOW}/<timestamp>/)"
FLOW_PROVIDER_VAR="FLOW${LLM_FLOW}_LLM_PROVIDER"
FLOW_MODEL_VAR="FLOW${LLM_FLOW}_LLM_MODEL"
EFFECTIVE_PROVIDER="${!FLOW_PROVIDER_VAR:-${LLM_PROVIDER}}"
EFFECTIVE_MODEL="${!FLOW_MODEL_VAR:-${LLM_MODEL}}"
echo "  LLM (global default)  : ${LLM_PROVIDER}/${LLM_MODEL}"
echo "  LLM (effective flow)  : ${EFFECTIVE_PROVIDER}/${EFFECTIVE_MODEL}"
if [[ -n "$ENV_FILE" ]]; then
  echo "  .env loaded from      : ${ENV_FILE}"
fi
echo "──────────────────────────────────────────────────────────────────────"

LAUNCH_ARGS=(
  "map_yaml:=${MAP_YAML_PATH}"
  "source_x:=${SOURCE_X}"
  "source_y:=${SOURCE_Y}"
  "source_yaw:=${SOURCE_YAW}"
  "dest_x:=${DEST_X}"
  "dest_y:=${DEST_Y}"
  "dest_yaw:=${DEST_YAW}"
  "spawn_x:=${SPAWN_X}"
  "spawn_y:=${SPAWN_Y}"
  "spawn_z:=${SPAWN_Z}"
  "spawn_yaw:=${SPAWN_YAW}"
  "static_tf_x:=${STATIC_TF_X}"
  "static_tf_y:=${STATIC_TF_Y}"
  "use_sim_time:=${USE_SIM_TIME}"
  "launch_rviz:=${LAUNCH_RVIZ}"
  "flow:=${LLM_FLOW}"
)

if [[ "${HAS_WORLD}" == "1" ]]; then
  LAUNCH_ARGS+=("world:=${WORLD_FILE}")
fi

if [[ -n "${GAZEBO_MODEL_PATH:-}" ]]; then
  LAUNCH_ARGS+=("gazebo_model_path:=${GAZEBO_MODEL_PATH}")
fi

exec ros2 launch nav2_llm_demo llm_agent.launch.py "${LAUNCH_ARGS[@]}"
