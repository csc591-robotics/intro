# Creating a custom map (`custom_map.pgm` + `custom_map.yaml`)

This guide explains how to build a new occupancy grid map for Nav2 using **SLAM** (Simultaneous Localization and Mapping) in the **TurtleBot3 Gazebo** simulation. The result is a pair of files:

- **`*.pgm`** — grayscale image of the grid (free / occupied / unknown cells)
- **`*.yaml`** — metadata: image path, `resolution`, `origin`, thresholds

The `intro` demo loads maps such as `src/nav2_llm_demo/config/custom_map.yaml`, which points at the matching `.pgm`.

---

## Prerequisites

- ROS 2 **Humble** sourced: `source /opt/ros/humble/setup.bash`
- TurtleBot3 simulation packages (e.g. `ros-humble-turtlebot3-gazebo`, `ros-humble-turtlebot3-*` as needed for your install)
- A SLAM stack: **slam_toolbox** and/or **turtlebot3_cartographer** (Cartographer)
- **nav2_map_server** (for `map_saver_cli`)
- **Teleop** (see below) or any node that publishes **`geometry_msgs/msg/Twist`** on **`/cmd_vel`**

---

## Set `TURTLEBOT3_MODEL` (required for TurtleBot3 launches)

TurtleBot3 launch files read the **`TURTLEBOT3_MODEL`** environment variable. If it is unset, you may see launch errors such as `'TURTLEBOT3_MODEL'`.

Use the same model as your robot URDF/world expects. For the default TurtleBot3 world in this project, **`burger`** is typical:

```bash
export TURTLEBOT3_MODEL=burger
```

Other common values: `waffle`, `waffle_pi`.

You can put this in your shell profile, or prefix a single command:

```bash
TURTLEBOT3_MODEL=burger ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py
```

`run_llm_nav.sh` in this repo exports `TURTLEBOT3_MODEL` with default `burger` after sourcing `.env` (see `run_llm_nav.sh`).

---

## Install teleop (`turtlebot3_teleop`)

If `ros2 run turtlebot3_teleop teleop_keyboard` reports **Package 'turtlebot3_teleop' not found**, install the package (Ubuntu / Debian, ROS 2 Humble):

```bash
sudo apt update
sudo apt install ros-humble-turtlebot3-teleop
```

Then source ROS again and ensure `TURTLEBOT3_MODEL` is set before running teleop:

```bash
source /opt/ros/humble/setup.bash
export TURTLEBOT3_MODEL=burger
ros2 run turtlebot3_teleop teleop_keyboard
```

**Alternative** (if you prefer not to use TurtleBot3 teleop): generic keyboard teleop, if installed:

```bash
sudo apt install ros-humble-teleop-twist-keyboard
source /opt/ros/humble/setup.bash
ros2 run teleop_twist_keyboard teleop_twist_keyboard
```

Any method that drives the robot so **LiDAR** (`/scan`) sees the walls is enough for mapping.

---

## SLAM workflow (simulation)

Use **`use_sim_time:=true`** for all nodes while Gazebo is the clock source, so SLAM and the robot stay time-synchronized.

### Step 1 — Launch Gazebo with the TurtleBot3

In terminal 1:

```bash
source /opt/ros/humble/setup.bash
export TURTLEBOT3_MODEL=burger
ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py
```

### Step 2 — Launch SLAM

In terminal 2, pick **one** SLAM stack.

**Option A — slam_toolbox**

```bash
source /opt/ros/humble/setup.bash
ros2 launch slam_toolbox online_async_launch.py use_sim_time:=true
```

**Option B — Cartographer (TurtleBot3 example)**

```bash
source /opt/ros/humble/setup.bash
export TURTLEBOT3_MODEL=burger
ros2 launch turtlebot3_cartographer cartographer.launch.py use_sim_time:=True
```

Note: some launch files use `True` vs `true`; both are usually accepted for boolean launch args.

### Step 3 — Drive the robot so LiDAR covers the environment

In terminal 3:

```bash
source /opt/ros/humble/setup.bash
export TURTLEBOT3_MODEL=burger
ros2 run turtlebot3_teleop teleop_keyboard
```

Drive slowly through free space until walls and openings you care about are well covered.

### Step 4 — Save the map

When coverage is good, in another terminal:

```bash
source /opt/ros/humble/setup.bash
ros2 run nav2_map_server map_saver_cli -f custom_map
```

This writes **`custom_map.pgm`** and **`custom_map.yaml`** in the **current working directory** unless you pass a path in `-f`.

Copy them into your package config, e.g. `intro/src/nav2_llm_demo/config/`, and point Nav2 at the `.yaml` (as `run_llm_nav.sh` does with `NAV2_MAP` / `custom_map.yaml`).

---

## Tips

- **Origin and resolution** in the saved `.yaml` come from the SLAM / map frame used during mapping; if you change worlds or spawn pose, re-map or adjust `origin` carefully so AMCL and Nav2 stay consistent.
- **Unknown (gray)** cells mean “never observed”; drive so important areas are **white** (free) where the robot should plan.
- For headless servers, mapping still works as long as `/scan` and TF are valid and `use_sim_time` is consistent.

---

## See also

- [run_llm_nav_phases.md](run_llm_nav_phases.md) — how this repo launches Gazebo, Nav2, and the LLM node
- [architecture.md](architecture.md) — system overview
