#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# world_to_map runner.
#
# Usage:
#   bash intro/src/world_to_map/run_world_to_map.sh <map_name> [extra args]
#
# <map_name> is matched (in order) against:
#   1. intro/world_files/<name>.world
#   2. intro/world_files/gazebo_models_worlds_collection/worlds/<name>.world
#
# Run with no args to list every world we know about.
#
# Examples:
#   bash intro/src/world_to_map/run_world_to_map.sh diamond_map
#   bash intro/src/world_to_map/run_world_to_map.sh workshop_example
#   bash intro/src/world_to_map/run_world_to_map.sh house
#
# Optional environment overrides:
#   RESOLUTION   meters per pixel (default 0.05)
#   PADDING      empty border (m) around world AABB (default 1.0)
#   Z_MIN, Z_MAX robot height band (m) for occupancy filtering
#   X_POSE,Y_POSE,YAW  TurtleBot3 spawn pose in Gazebo world frame
#   ORIGIN_MODE  bottom-left (default) | world
#   FLOOR_CLEARANCE  skip box collisions whose top is <= this height (m).
#                Default 0.05. Filters Workshop-style floor slabs that would
#                otherwise fill the room interior with one giant black blob.
#   FORCE=1      regenerate the map even if it already exists
#   LAUNCH_RVIZ  true (default) | false
#   EXTRA_GAZEBO_MODEL_PATH  prepend additional model dirs (colon-separated)
#
# After this script reports "Streaming Gazebo + RViz", open a SECOND terminal
# in the same container and run the bundled WASDX teleop (no extra apt
# packages required):
#
#   docker compose exec autonomous_pathing_llm bash
#   export TURTLEBOT3_MODEL=burger
#   source /opt/ros/humble/setup.bash && source install/setup.bash
#   ros2 run world_to_map teleop_wasdx
#
# (controls: w / x = forward / back, a / d = turn left / right, s = stop,
#  q = quit, +/- linear step, ]/[ angular step)
# ---------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_DIR="$SCRIPT_DIR"
WORKSPACE_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
WORLD_DIR_PRIMARY="$WORKSPACE_DIR/world_files"
COLLECTION_DIR="$WORLD_DIR_PRIMARY/gazebo_models_worlds_collection"
WORLD_DIR_COLLECTION="$COLLECTION_DIR/worlds"
MODEL_DIR_COLLECTION="$COLLECTION_DIR/models"
MAPS_DIR="$PKG_DIR/maps"

list_worlds() {
  echo "Worlds in $WORLD_DIR_PRIMARY:"
  for f in "$WORLD_DIR_PRIMARY"/*.world; do
    [[ -e "$f" ]] || continue
    echo "  - $(basename "$f" .world)"
  done
  if [[ -d "$WORLD_DIR_COLLECTION" ]]; then
    echo ""
    echo "Worlds in $WORLD_DIR_COLLECTION:"
    for f in "$WORLD_DIR_COLLECTION"/*.world; do
      [[ -e "$f" ]] || continue
      echo "  - $(basename "$f" .world)"
    done
  fi
}

if [[ $# -lt 1 ]]; then
  echo "Usage: bash $0 <map_name>"
  echo ""
  list_worlds
  exit 1
fi

RAW_NAME="$1"; shift || true
MAP_NAME="${RAW_NAME%.world}"

WORLD_FILE=""
for candidate in "$WORLD_DIR_PRIMARY/${MAP_NAME}.world" \
                 "$WORLD_DIR_COLLECTION/${MAP_NAME}.world"; do
  if [[ -f "$candidate" ]]; then
    WORLD_FILE="$candidate"
    break
  fi
done
if [[ -z "$WORLD_FILE" ]]; then
  echo "World file not found for: $MAP_NAME"
  echo ""
  list_worlds
  exit 1
fi

RESOLUTION="${RESOLUTION:-0.05}"
PADDING="${PADDING:-2.0}"
Z_MIN="${Z_MIN:-0.0}"
Z_MAX="${Z_MAX:-0.4}"
X_POSE="${X_POSE:-0.0}"
Y_POSE="${Y_POSE:-0.0}"
Z_POSE="${Z_POSE:-0.05}"
YAW="${YAW:-0.0}"
ORIGIN_MODE="${ORIGIN_MODE:-bottom-left}"
LAUNCH_RVIZ="${LAUNCH_RVIZ:-true}"
FORCE="${FORCE:-0}"
FLOOR_CLEARANCE="${FLOOR_CLEARANCE:-0.05}"
EXTRA_GAZEBO_MODEL_PATH="${EXTRA_GAZEBO_MODEL_PATH:-}"
SCHEMA_REQUIRED=3

MODEL_PATHS=""
if [[ -d "$MODEL_DIR_COLLECTION" ]]; then
  MODEL_PATHS="$MODEL_DIR_COLLECTION"
fi
if [[ -n "$EXTRA_GAZEBO_MODEL_PATH" ]]; then
  if [[ -n "$MODEL_PATHS" ]]; then
    MODEL_PATHS="$EXTRA_GAZEBO_MODEL_PATH:$MODEL_PATHS"
  else
    MODEL_PATHS="$EXTRA_GAZEBO_MODEL_PATH"
  fi
fi

mkdir -p "$MAPS_DIR"
OUT_PREFIX="$MAPS_DIR/$MAP_NAME"
PGM="$OUT_PREFIX.pgm"
YAML="$OUT_PREFIX.yaml"
SIDECAR="${OUT_PREFIX}.world_map.yaml"

cd "$WORKSPACE_DIR"

set +u
source /opt/ros/humble/setup.bash
set -u

if [[ ! -f install/setup.bash ]]; then
  echo "Missing install/setup.bash. Build the workspace first:"
  echo "  colcon build --packages-select world_to_map"
  exit 1
fi

set +u
source install/setup.bash
set -u

export TURTLEBOT3_MODEL="${TURTLEBOT3_MODEL:-burger}"
if [[ -n "$MODEL_PATHS" ]]; then
  if [[ -n "${GAZEBO_MODEL_PATH:-}" ]]; then
    export GAZEBO_MODEL_PATH="$MODEL_PATHS:$GAZEBO_MODEL_PATH"
  else
    export GAZEBO_MODEL_PATH="$MODEL_PATHS"
  fi
fi

echo "World file       : $WORLD_FILE"
echo "Map name         : $MAP_NAME"
echo "Origin mode      : $ORIGIN_MODE"
echo "Model paths      : ${GAZEBO_MODEL_PATH:-<none>}"

NEED_REGEN=0
if [[ "$FORCE" == "1" ]]; then
  NEED_REGEN=1
elif [[ ! -f "$PGM" || ! -f "$YAML" || ! -f "$SIDECAR" ]]; then
  NEED_REGEN=1
else
  CURRENT_SCHEMA="$(python3 -c "import yaml; d=yaml.safe_load(open('$SIDECAR')); print(d.get('schema_version',0))" 2>/dev/null || echo 0)"
  if [[ "$CURRENT_SCHEMA" -lt "$SCHEMA_REQUIRED" ]]; then
    echo "Existing map sidecar is schema v$CURRENT_SCHEMA (need v$SCHEMA_REQUIRED). Regenerating."
    NEED_REGEN=1
  fi
fi

if [[ "$NEED_REGEN" == "1" ]]; then
  echo "Rasterizing $WORLD_FILE -> $OUT_PREFIX.{pgm,yaml,world_map.yaml}"
  rm -f "$PGM" "$YAML" "$SIDECAR" 2>/dev/null || true
  ros2 run world_to_map rasterize_world \
    --world "$WORLD_FILE" \
    --out "$OUT_PREFIX" \
    --resolution "$RESOLUTION" \
    --padding "$PADDING" \
    --z-min "$Z_MIN" \
    --z-max "$Z_MAX" \
    --floor-clearance "$FLOOR_CLEARANCE" \
    --origin-mode "$ORIGIN_MODE" \
    --include-point "$X_POSE" "$Y_POSE" \
    --model-paths "${GAZEBO_MODEL_PATH:-}"
else
  echo "Reusing existing map files (set FORCE=1 to regenerate):"
  echo "  $PGM"
  echo "  $YAML"
  echo "  $SIDECAR"
fi

OFFSET_X="$(python3 -c "import yaml,sys; d=yaml.safe_load(open('$SIDECAR')); print(d.get('world_to_map_offset',[0,0])[0])")"
OFFSET_Y="$(python3 -c "import yaml,sys; d=yaml.safe_load(open('$SIDECAR')); print(d.get('world_to_map_offset',[0,0])[1])")"
echo "map->odom offset : ($OFFSET_X, $OFFSET_Y)"

echo ""
echo "----------------------------------------------------------------------"
echo "In a SECOND terminal, run the bundled WASDX teleop:"
echo ""
echo "  export TURTLEBOT3_MODEL=$TURTLEBOT3_MODEL"
echo "  source /opt/ros/humble/setup.bash && source install/setup.bash"
echo "  ros2 run world_to_map teleop_wasdx"
echo ""
echo "  controls: w/x = forward/back, a/d = turn, s = stop, q = quit"
echo "----------------------------------------------------------------------"
echo ""

ros2 launch world_to_map world_to_map.launch.py \
  world:="$WORLD_FILE" \
  map_yaml:="$YAML" \
  x_pose:="$X_POSE" \
  y_pose:="$Y_POSE" \
  z_pose:="$Z_POSE" \
  yaw:="$YAW" \
  launch_rviz:="$LAUNCH_RVIZ" \
  map_offset_x:="$OFFSET_X" \
  map_offset_y:="$OFFSET_Y" \
  gazebo_model_path:="${GAZEBO_MODEL_PATH:-}"
