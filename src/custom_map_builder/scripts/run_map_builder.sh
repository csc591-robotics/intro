#!/usr/bin/env bash
# Run after: cd intro && source install/setup.bash
# Optional first argument: path or filename of map YAML (see package maps/ folder).

set -euo pipefail

export TURTLEBOT3_MODEL="${TURTLEBOT3_MODEL:-burger}"

if [[ $# -ge 1 ]]; then
  exec ros2 launch custom_map_builder map_builder.launch.py "map_yaml:=$1"
else
  exec ros2 launch custom_map_builder map_builder.launch.py
fi
