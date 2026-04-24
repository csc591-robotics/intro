## Working Startup

Run `run_llm_nav.sh` only inside the Docker container.
Do not run it from your Mac host shell.

## Warehouse

Use this when the full `warehouse.world` assets are not available and you want
the warehouse map + graph agent to run reliably.

### Terminal 1 (host): start container
```bash
docker compose up -d --build
docker compose exec autonomous_pathing_llm bash
```

### Terminal 2 (container): build + source
```bash
cd /workspace
source /workspace/.env
source /opt/ros/humble/setup.bash
rm -rf build/nav2_llm_demo install/nav2_llm_demo log
colcon build --packages-select nav2_llm_demo
source /workspace/install/setup.bash
```

### Terminal 3 (container): run simulator (empty world, headless)
```bash
cd /workspace
source /workspace/.env
source /opt/ros/humble/setup.bash
source /workspace/install/setup.bash
export TURTLEBOT3_MODEL=burger
ros2 launch turtlebot3_gazebo empty_world.launch.py x_pose:=3.981274 y_pose:=7.866986 gui:=false
```

### Terminal 4 (container): run warehouse map + graph node
```bash
cd /workspace
source /workspace/.env
source /opt/ros/humble/setup.bash
source /workspace/install/setup.bash
ros2 launch nav2_llm_demo llm_agent.launch.py \
  map_yaml:=/workspace/src/world_to_map/maps/warehouse.yaml \
  source_x:=3.981274 source_y:=7.866986 source_yaw:=0.0 \
  dest_x:=11.512874 dest_y:=13.666186 dest_yaw:=0.0 \
  static_tf_x:=0.0 static_tf_y:=0.0 \
  use_sim_time:=true launch_rviz:=false
```

### Terminal 5 (container): quick checks
```bash
cd /workspace
source /workspace/.env
source /opt/ros/humble/setup.bash
source /workspace/install/setup.bash
ros2 topic echo /odom --once
python3 - <<'PY'
import json
g = json.load(open('/workspace/topology_graph_debug.json'))
print('nodes=', len(g['nodes']), 'edges=', len(g['edges']))
print('node_ids=', [n['node_id'] for n in g['nodes']])
print('edges=', [(e['from_node'], e['to_node']) for e in g['edges']])
PY
```


## In a terminal on your local machine
```bash
xhost +local:docker
```

## Terminal 1: Start Docker
From the repo root on your Mac:

```bash
docker compose up -d --build
```

## Terminal 2: Build The ROS Package

Open a new terminal on your Mac, then enter the container:

```bash
docker compose exec autonomous_pathing_llm bash
```

Inside the container:

```bash
source /opt/ros/humble/setup.bash
colcon build --packages-select nav2_llm_demo
source install/setup.bash
```

If the build succeeds, stay in this terminal if you want, or exit it.

## Terminal 3: Run the LLM Agent

Open a new terminal on your Mac, then enter the container:

```bash
docker compose exec autonomous_pathing_llm bash
```

Inside the container, run with a map name:

```bash
bash ./run_llm_nav.sh diamond_blocked
```

The map name must match an entry in `src/nav2_llm_demo/maps/nav_config.yaml`.

That script does all of this:
- parses `nav_config.yaml` for the map's source/destination poses
- starts Gazebo (empty world) with TurtleBot3 spawned at the source pose
- starts `map_server` with the selected map
- starts the `llm_agent_node` (LangGraph vision agent)
- streams `/navigation_status` updates

The agent will:
1. Capture a top-down annotated map view
2. Use the vision LLM to decide movement actions
3. Control the robot via `cmd_vel` (move forward, rotate)
4. Repeat until it reaches the destination

Leave this terminal open while the system is running.

## Adding a New Map

1. Create/test the map in `custom_map_builder` (using `run_map_builder.sh`).
2. Copy the `.pgm` and `.yaml` files into `src/nav2_llm_demo/maps/`.
3. Add an entry in `src/nav2_llm_demo/maps/nav_config.yaml` with source/destination poses.
4. Rebuild: `colcon build --packages-select nav2_llm_demo`

## If It Fails Immediately

Check these first:
- you are inside the Docker container, not your Mac host shell
- `.env` exists at `/workspace/.env`
- `.env` contains `LLM_PROVIDER=...` and `LLM_MODEL=...` plus the matching provider API key (see `.env.example`)
- the LLM model must support **vision** (e.g. `gpt-4o`, `claude-3-5-sonnet-20241022`, `gemini-1.5-pro`)
- `colcon build --packages-select nav2_llm_demo` finished successfully
- the map files (`.pgm` + `.yaml`) exist in `src/nav2_llm_demo/maps/`
- you left Terminal 3 open after starting `run_llm_nav.sh`

## Gazebo GUI: `Authorization required` / `gzclient` died

`run_llm_nav.sh` starts the full Gazebo stack, including **`gzclient`** (the 3D window). That only works if your shell has a **working display** (X11) and permission to use it.

If you see `Authorization required, but no authorization protocol specified` and `[ERROR] [gzclient-2]: process has died`, the physics server (`gzserver`) may still be running, but the **GUI cannot open**. Typical causes:

- SSH without forwarding: plain `ssh user@host` has no display. Use **X11 forwarding** (`ssh -Y user@host`) **and** an X server on your laptop (XQuartz on macOS, VcXsrv/WSLg on Windows), **or**
- **Remote VM / lab machine (e.g. VCL):** run `run_llm_nav.sh` from a **desktop session** for that VM (VNC, noVNC, or the provider's "console" GUI), not only from a text-only SSH session. Open a terminal **inside** that desktop so `DISPLAY` is set (often `:0` or `:1`).
- **Wrong `DISPLAY` or missing cookie:** the user that starts Gazebo must match the user logged into the graphical session, or you must **merge `xauth`** / use the same `DISPLAY` as the active desktop.

There is no project setting that fixes this; the fix is to run Gazebo where a real (or forwarded) X session exists. `gzserver` alone does not need a monitor, but **you asked for the full Gazebo window**, so the environment must provide one.

### Docker on Linux: allow the container to use your X11 display

`docker-compose.yaml` already passes `DISPLAY` and mounts `/tmp/.X11-unix`. The host X server still has to **accept connections** from processes inside the container (otherwise you get `Authorization required, but no authorization protocol specified`).

On the **Linux host**, in the same graphical session where your desktop runs, run **once** before `docker compose up` (or after each login):

```bash
xhost +local:docker
```

If your X server reports that `docker` is not a valid family name, use the broader local rule:

```bash
xhost +local:
```

(`+local:` allows any local user to connect to your display; only use on trusted machines.)

To tighten access when you are done:

```bash
xhost -local:docker
# or
xhost -local:
```

Ensure `echo $DISPLAY` is set on the host when you start Compose (e.g. `:0` or `:1`).

### macOS + Docker Desktop

`xhost` on the Mac does not apply the same way. You typically need **XQuartz**, configure it to accept network connections, and point the container at your host's display (Docker Desktop networking differs from Linux `network_mode: host`). Follow a ROS-on-Docker + XQuartz guide if you run the GUI from a container on Mac.



  rm -rf build/nav2_llm_demo install/nav2_llm_demo log
  source /opt/ros/humble/setup.bash
  colcon build --packages-select nav2_llm_demo
  source install/setup.bash
  bash ./run_llm_nav.sh diamond_blocked
