#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# custom_map_builder runner — loads a PGM map in RViz for clicking
# start / destination poses.
#
# Usage (inside the Docker container, after building):
#
#   cd /workspace/intro
#   source install/setup.bash
#
#   # Any hand-crafted map (no offset needed):
#   bash src/custom_map_builder/scripts/run_map_builder.sh diamond_blocked
#
#   # A world_to_map-generated map (offset auto-read from .world_map.yaml):
#   bash src/custom_map_builder/scripts/run_map_builder.sh \
#        src/world_to_map/maps/warehouse.yaml
#
# The runner looks for a sibling <name>.world_map.yaml sidecar next to the
# YAML file.  If found, world_to_map_offset is read from it and passed as
# map_offset_x / map_offset_y so the map→odom TF aligns map-frame clicks
# with Gazebo world coordinates.
#
# Output while clicking in RViz (Publish Point tool):
#   frame="map"  MAP  x=...  y=...  |  GAZEBO WORLD  x=...  y=...
#
# Optional environment overrides:
#   LAUNCH_GAZEBO  true (default) | false  – start Gazebo alongside RViz
#   LAUNCH_RVIZ    true (default) | false  – start RViz
# ---------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"

LAUNCH_GAZEBO="${LAUNCH_GAZEBO:-true}"
LAUNCH_RVIZ="${LAUNCH_RVIZ:-true}"

if [[ $# -lt 1 ]]; then
  echo "Usage: bash $0 <map_name_or_yaml_path>"
  echo ""
  echo "Examples:"
  echo "  bash $0 diamond_blocked"
  echo "  bash $0 src/world_to_map/maps/warehouse.yaml"
  exit 1
fi

RAW_ARG="$1"

# ── Resolve the YAML path ──────────────────────────────────────────────────
YAML_PATH=""
if [[ "$RAW_ARG" == *.yaml ]] || [[ "$RAW_ARG" == *.yml ]]; then
  if [[ -f "$RAW_ARG" ]]; then
    YAML_PATH="$(realpath "$RAW_ARG")"
  elif [[ -f "$WORKSPACE_DIR/$RAW_ARG" ]]; then
    YAML_PATH="$(realpath "$WORKSPACE_DIR/$RAW_ARG")"
  fi
else
  # Bare name: strip .pgm/.yaml suffix if present
  MAP_NAME="${RAW_ARG%.yaml}"
  MAP_NAME="${MAP_NAME%.pgm}"
  # Search order: world_to_map maps, then custom_map_builder maps
  for CANDIDATE in \
    "$WORKSPACE_DIR/src/world_to_map/maps/${MAP_NAME}.yaml" \
    "$WORKSPACE_DIR/src/custom_map_builder/maps/${MAP_NAME}.yaml"; do
    if [[ -f "$CANDIDATE" ]]; then
      YAML_PATH="$(realpath "$CANDIDATE")"
      break
    fi
  done
fi

if [[ -z "$YAML_PATH" ]]; then
  echo "ERROR: could not find a .yaml map for: $RAW_ARG"
  echo "Tried:"
  echo "  $WORKSPACE_DIR/src/world_to_map/maps/<name>.yaml"
  echo "  $WORKSPACE_DIR/src/custom_map_builder/maps/<name>.yaml"
  exit 1
fi

echo "Map YAML         : $YAML_PATH"

# ── Auto-read world_to_map_offset from sidecar if present ─────────────────
SIDECAR="${YAML_PATH%.yaml}.world_map.yaml"
MAP_OFFSET_X="0.0"
MAP_OFFSET_Y="0.0"
WORLD_FILE=""
if [[ -f "$SIDECAR" ]]; then
  echo "Sidecar          : $SIDECAR"
  SIDECAR_INFO="$(python3 - "$SIDECAR" <<'PYEOF'
import sys, yaml
with open(sys.argv[1]) as fh:
    d = yaml.safe_load(fh)
off = d.get('world_to_map_offset', [0.0, 0.0])
world = d.get('source_world', '')
print(f"{float(off[0]):.6f}|{float(off[1]):.6f}|{world}")
PYEOF
)"
  MAP_OFFSET_X="${SIDECAR_INFO%%|*}"
  REST="${SIDECAR_INFO#*|}"
  MAP_OFFSET_Y="${REST%%|*}"
  WORLD_FILE="${REST#*|}"
  echo "map→odom offset  : ($MAP_OFFSET_X, $MAP_OFFSET_Y)"
  echo "  → click coords in RViz will show both map-frame AND Gazebo world coords"
  if [[ -n "$WORLD_FILE" ]]; then
    if [[ ! -f "$WORLD_FILE" ]]; then
      echo "WARNING: source_world from sidecar does not exist: $WORLD_FILE"
      echo "         Falling back to default TurtleBot3 world."
      WORLD_FILE=""
    else
      echo "World file       : $WORLD_FILE"
    fi
  fi
else
  echo "No sidecar found → map_offset = (0, 0)  (fine for hand-crafted maps)"
fi

# Auto-detect gazebo_models_worlds_collection so meshed worlds (e.g.
# warehouse) can resolve their model:// URIs.
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

# ── Source ROS ────────────────────────────────────────────────────────────
cd "$WORKSPACE_DIR"
set +u
source /opt/ros/humble/setup.bash
if [[ -f install/setup.bash ]]; then
  source install/setup.bash
else
  echo "WARNING: install/setup.bash not found. Build the workspace first:"
  echo "  colcon build --packages-select custom_map_builder"
fi
set -u

export TURTLEBOT3_MODEL="${TURTLEBOT3_MODEL:-burger}"

echo ""
echo "----------------------------------------------------------------------"
echo "In RViz, select the 'Publish Point' tool (shortcut: P) and click"
echo "anywhere on the map.  The terminal will print:"
echo "  frame=\"map\"  MAP  x=<map_x>  y=<map_y>  |  GAZEBO WORLD  x=<wx>  y=<wy>"
echo ""
echo "Use the GAZEBO WORLD coordinates as source/destination for nav2."
echo "----------------------------------------------------------------------"
echo ""

LAUNCH_ARGS=(
  "map_yaml:=$YAML_PATH"
  "map_offset_x:=$MAP_OFFSET_X"
  "map_offset_y:=$MAP_OFFSET_Y"
  "launch_gazebo:=$LAUNCH_GAZEBO"
  "launch_rviz:=$LAUNCH_RVIZ"
)
if [[ -n "$WORLD_FILE" ]]; then
  LAUNCH_ARGS+=("world:=$WORLD_FILE")
fi
if [[ -n "$GAZEBO_MODEL_PATH" ]]; then
  LAUNCH_ARGS+=("gazebo_model_path:=$GAZEBO_MODEL_PATH")
fi

ros2 launch custom_map_builder map_builder.launch.py "${LAUNCH_ARGS[@]}"
