## Working Startup

Run `run_llm_nav.sh` only inside the Docker container.
Do not run it from your Mac host shell.


## In a terminal on your local machine
```bash
xhost +local:docker
```

## Terminal 1: Start Docker
From the repo root on your Mac:

```bash
docker compose up -d --build
```
s
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

## Terminal 3: Run Everything For Step 1-4

Open a new terminal on your Mac, then enter the container:

```bash
docker compose exec autonomous_pathing_llm bash
```

Inside the container:

```bash
bash ./run_llm_nav.sh
```

Or with a custom mission:

```bash
bash ./run_llm_nav.sh "Reach the far side of the obstacle course"
```

That script now does all of this:
- starts Gazebo + TurtleBot3
- starts Nav2 + AMCL
- starts `llm_nav_node`
- starts a background `ros2 topic echo /navigation_status`
- publishes one mission request to `/navigation_request`

Leave this terminal open while the system is running.

## Terminal 4: Optional Manual Mission Publish

Only open this if you want to send another mission while Terminal 3 is still running.

On your Mac:

```bash
docker compose exec autonomous_pathing_llm bash
```

Inside the container:

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 topic pub --once /navigation_request std_msgs/msg/String "{data: 'Reach the loading zone'}"
```

## If It Fails Immediately

Check these first:
- you are inside the Docker container, not your Mac host shell
- `.env` exists at `/workspace/.env`
- `.env` contains `LLM_PROVIDER=...` and `LLM_MODEL=...` plus the matching provider API key (see `.env.example`)
- `colcon build --packages-select nav2_llm_demo` finished successfully
- you left Terminal 3 open after starting `run_llm_nav.sh`

## Gazebo GUI: `Authorization required` / `gzclient` died

`run_llm_nav.sh` starts the full Gazebo stack, including **`gzclient`** (the 3D window). That only works if your shell has a **working display** (X11) and permission to use it.

If you see `Authorization required, but no authorization protocol specified` and `[ERROR] [gzclient-2]: process has died`, the physics server (`gzserver`) may still be running, but the **GUI cannot open**. Typical causes:

- SSH without forwarding: plain `ssh user@host` has no display. Use **X11 forwarding** (`ssh -Y user@host`) **and** an X server on your laptop (XQuartz on macOS, VcXsrv/WSLg on Windows), **or**
- **Remote VM / lab machine (e.g. VCL):** run `run_llm_nav.sh` from a **desktop session** for that VM (VNC, noVNC, or the provider’s “console” GUI), not only from a text-only SSH session. Open a terminal **inside** that desktop so `DISPLAY` is set (often `:0` or `:1`).
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

`xhost` on the Mac does not apply the same way. You typically need **XQuartz**, configure it to accept network connections, and point the container at your host’s display (Docker Desktop networking differs from Linux `network_mode: host`). Follow a ROS-on-Docker + XQuartz guide if you run the GUI from a container on Mac.

## Rasterize a .world into an RViz map (`world_to_map` package)

Inside the container:

```bash
cd /workspace/intro
colcon build --packages-select world_to_map
source install/setup.bash
# List every world the runner knows about (originals + collection).
bash src/world_to_map/run_world_to_map.sh
# Pick one — examples below:
bash src/world_to_map/run_world_to_map.sh diamond_map
bash src/world_to_map/run_world_to_map.sh workshop_example
bash src/world_to_map/run_world_to_map.sh house
```

The runner also searches `intro/world_files/gazebo_models_worlds_collection/worlds/`
and points `GAZEBO_MODEL_PATH` at that collection's `models/` so every
`model://...` include resolves both for Gazebo (visual render) and for the
rasterizer (RViz occupancy grid).

It generates `intro/src/world_to_map/maps/<name>.{pgm,yaml,world_map.yaml}`
and starts Gazebo + map_server + RViz with a TurtleBot3 spawned at the
origin. In a SECOND terminal, drive it with the bundled WASDX teleop
(`w`/`x` forward-back, `a`/`d` turn, `s` stop, `q` quit):

```bash
docker compose exec autonomous_pathing_llm bash
export TURTLEBOT3_MODEL=burger
source /opt/ros/humble/setup.bash && source install/setup.bash
ros2 run world_to_map teleop_wasdx
```

By default the rasterized map sits in the **positive quadrant** with its
bottom-left corner at the map frame origin (0, 0), and the launch
publishes a static `map -> odom` transform with the matching offset so
2 m driven in Gazebo still shows as 2 m on the RViz map. Set
`ORIGIN_MODE=world` to make `map` frame identical to Gazebo's `world`
frame instead. See `src/world_to_map/README.md` for all env overrides
(`RESOLUTION`, `PADDING`, `Z_MIN/MAX`, `X_POSE/Y_POSE/YAW`, `ORIGIN_MODE`,
`FORCE`, `EXTRA_GAZEBO_MODEL_PATH`).

## Step 2: Start LangGraph controller
docker compose exec autonomous_pathing_llm bash
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch nav2_llm_tools step2_tools.launch.py

## Step 2: Start MCP server for an external LLM client
docker compose exec autonomous_pathing_llm bash
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 run nav2_llm_tools mcp_server
