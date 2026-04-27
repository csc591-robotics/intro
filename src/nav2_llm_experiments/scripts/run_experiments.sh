#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────
# run_experiments.sh — batch experiment runner on top of nav2_llm_demo.
#
# Reads src/nav2_llm_experiments/config/experiments.yaml and runs each
# selected experiment for each selected flow. Each (experiment, flow)
# combo spins up a fresh Gazebo + RViz + agent (just like
# `bash run_llm_nav.sh <map> --flow <N>` would), records LLM calls /
# pose / scan / cmd_vel / actions / rosbag2, then tears everything down
# before moving to the next combo.
#
# Usage:
#   bash src/nav2_llm_experiments/scripts/run_experiments.sh \
#       [--config <path>] \
#       [--map-poses <path>] \
#       [--flow 1,2,3,4,5,6,7] \
#       [--experiment 1,3,5] \
#       [--output-dir <path>] \
#       [--no-rosbag] \
#       [--no-rviz] \
#       [--no-gazebo-gui]
#
# Examples:
#   # Run every experiment for flow 5 (default)
#   bash src/nav2_llm_experiments/scripts/run_experiments.sh
#
#   # Run all experiments, both flow 3 and flow 5 (cross product)
#   bash src/nav2_llm_experiments/scripts/run_experiments.sh --flow 3,5
#
#   # Just experiments 1 and 5, flow 5, no RViz, no rosbag
#   bash src/nav2_llm_experiments/scripts/run_experiments.sh \
#       --experiment 1,5 --flow 5 --no-rviz --no-rosbag
#
#   # Pure Nav2 baseline (no LLM) for every experiment
#   bash src/nav2_llm_experiments/scripts/run_experiments.sh --flow 6
#
#   # LLM route planner over a topology graph (flow 7)
#   bash src/nav2_llm_experiments/scripts/run_experiments.sh --flow 7
# ──────────────────────────────────────────────────────────────────────────

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

# Source ROS unconditionally. Falling back is critical — when this script
# is launched from a `sudo su` shell or a minimal subshell, ROS_DISTRO may
# be empty even though /opt/ros/<distro>/setup.bash exists. Without sourcing,
# the orchestrator's child `ros2 run` / `ros2 bag` processes inherit a PATH
# and PYTHONPATH that don't know about ros2cli, and they crash with
# `No package metadata was found for ros2cli`.
ROS_DISTRO_CANDIDATES=("${ROS_DISTRO:-}" humble jazzy iron rolling galactic foxy)
for cand in "${ROS_DISTRO_CANDIDATES[@]}"; do
  if [[ -n "$cand" && -f "/opt/ros/${cand}/setup.bash" ]]; then
    set +u
    # shellcheck disable=SC1090
    source "/opt/ros/${cand}/setup.bash"
    set -u
    export ROS_DISTRO="$cand"
    break
  fi
done

if [[ -f "${WORKSPACE_ROOT}/install/setup.bash" ]]; then
  set +u
  # shellcheck disable=SC1091
  source "${WORKSPACE_ROOT}/install/setup.bash"
  set -u
fi

# Source .env (LLM_PROVIDER, LLM_MODEL, *_API_KEY, etc.) so the values
# are inherited by every subprocess the orchestrator spawns. Same logic
# as run_llm_nav.sh.
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
  exit 1
fi

# Pre-flight: confirm the orchestrator entry point is installed.
ORCH_EXE="${WORKSPACE_ROOT}/install/nav2_llm_experiments/lib/nav2_llm_experiments/run_experiments"
if [[ ! -x "$ORCH_EXE" ]]; then
  # Fall back to python -m so this works before/without colcon build.
  echo "(orchestrator console_script not found at $ORCH_EXE — falling back to python module)"
  exec python3 -m nav2_llm_experiments.run_experiments "$@"
fi

exec "$ORCH_EXE" "$@"
