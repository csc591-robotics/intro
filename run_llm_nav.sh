#!/usr/bin/env bash
set -euo pipefail

cd /workspace
source /opt/ros/humble/setup.bash
source install/setup.bash

if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi

if [[ -z "${GROQ_API_KEY:-}" ]]; then
  echo "GROQ_API_KEY is not set. Add it to /workspace/.env before launching."
  exit 1
fi

exec ros2 launch nav2_llm_demo llm_nav.launch.py
