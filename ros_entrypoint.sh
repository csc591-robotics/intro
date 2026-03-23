#!/bin/bash
set -e

. "/opt/ros/${ROS_DISTRO}/setup.bash"

[ -f /workspace/install/setup.bash ] && . /workspace/install/setup.bash

exec "$@"
