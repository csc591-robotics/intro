# custom_map_builder

ROS 2 package for **loading your own occupancy map (PGM + YAML)** together with **Gazebo**, a **TurtleBot3 Burger**, and **RViz**. Use RViz’s **Publish Point** tool to click on the map; a small node prints **`(x, y, z)` in the clicked message’s frame** (set **Fixed Frame** to `map` in RViz so clicks are in **map** coordinates).

This lives under `intro/src/custom_map_builder/` as part of the `intro` workspace.

## Build

From the `intro/` workspace (where `src/` and `install/` live):

```bash
cd intro
source /opt/ros/humble/setup.bash
colcon build --packages-select custom_map_builder
source install/setup.bash
```

**Maps folder:** Everything under `src/custom_map_builder/maps/` (any `.pgm`, `.yaml`, or other regular files, including in subfolders) is **copied into `install/.../share/custom_map_builder/maps/` when you build** (`colcon build`). Add or change files there, then rebuild and `source install/setup.bash` so `ros2 launch` sees them. Hidden files and dot-directories are skipped.

## Run (Gazebo + map + RViz + click echo)

```bash
export TURTLEBOT3_MODEL=burger   # required for TurtleBot3 Gazebo
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch custom_map_builder map_builder.launch.py
```

Optional arguments:

| Argument | Default | Meaning |
|----------|---------|---------|
| `map_yaml` | `$(ros2 pkg prefix custom_map_builder)/share/custom_map_builder/maps/default.yaml` | Absolute path to your map YAML, **or** a filename inside `share/custom_map_builder/maps/` |
| `use_sim_time` | `true` | Set `false` only if you disable Gazebo (`launch_gazebo:=false`) |
| `launch_gazebo` | `true` | `false` = map + RViz + echo only (no sim) |
| `launch_rviz` | `true` | `false` = no RViz window |

### Coordinates only (no Gazebo, no TurtleBot3)

To **read map-frame \((x, y)\)** from clicks without starting the simulator or the Burger, disable Gazebo and turn off sim time (nothing publishes `/clock`):

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch custom_map_builder map_builder.launch.py \
  launch_gazebo:=false \
  use_sim_time:=false
```

You still get **`map_server`**, the static **`map` → `odom`** transform, **RViz**, and **`map_click_echo`**. No `TURTLEBOT3_MODEL` is required for this mode. Use the **Publish Point** steps in [Click coordinates](#click-coordinates) below.

Example with your own YAML (after it is installed under `share/.../maps/` or via an absolute path):

```bash
ros2 launch custom_map_builder map_builder.launch.py \
  launch_gazebo:=false \
  use_sim_time:=false \
  map_yaml:=default.yaml
```

With **Gazebo + robot** (default), point at a map by absolute path or by filename under `share/.../maps/`:

```bash
ros2 launch custom_map_builder map_builder.launch.py \
  map_yaml:=/full/path/to/my_map.yaml
```

```bash
ros2 launch custom_map_builder map_builder.launch.py map_yaml:=default.yaml
```

Helper script (after install):

```bash
bash $(ros2 pkg prefix custom_map_builder)/share/custom_map_builder/scripts/run_map_builder.sh
bash $(ros2 pkg prefix custom_map_builder)/share/custom_map_builder/scripts/run_map_builder.sh /path/to/map.yaml
```

## Click coordinates

1. Wait until the **Map** display shows your grid in RViz.
2. In RViz, select the **Publish Point** tool (toolbar).
3. Click on the map.
4. Watch the terminal running the launch file: node **`map_click_echo`** logs lines like  
   `clicked_point frame="map" x=... y=... z=...`  
   Those **`x` and `y`** are the **map-frame** coordinates (meters) for Nav2 / `route_graph.json` style poses (use `z` as 0 for 2D).

**Note:** The launch publishes a **static `map` → `odom` identity**. The Burger’s pose in Gazebo is **not** tied to your custom map geometry unless you build a matching Gazebo world. For **picking coordinates on the occupancy grid**, that is fine: clicks are still valid **map** frame points from the loaded YAML/PGM.

## Y-shaped (or other) maps

Generate a simple **Y-shaped** corridor PGM + YAML (writes into this package’s `maps/` next to the script):

```bash
python3 src/custom_map_builder/scripts/generate_y_corridor_map.py
```

Then copy or symlink the generated `y_corridor.yaml` / `y_corridor.pgm` into `maps/`, rebuild (so `share/.../maps/` updates), or pass an **absolute path** to `map_yaml`.

Shipped **default** map is a small empty room (`maps/default.pgm`) so the package runs without running the generator first.

## Troubleshooting

- **`TURTLEBOT3_MODEL` launch error:** Export before launching: `export TURTLEBOT3_MODEL=burger` (or use `run_map_builder.sh`, which sets a default).
- **`use_rviz` unknown argument:** Your `turtlebot3_world.launch.py` may not define that argument. Edit `launch/map_builder.launch.py` and remove the `'use_rviz': 'false'` entry from `launch_arguments`, or upgrade TurtleBot3 packages.

## Related docs

- [../docs/custom_map_slam.md](../docs/custom_map_slam.md) — SLAM workflow to record maps from the real TurtleBot3 world in simulation.
