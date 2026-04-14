# What `run_llm_nav.sh` does (phase by phase)

This document walks through `intro/run_llm_nav.sh` in execution order and relates each phase to what you typically see in the terminal logs.

For a higher-level system overview, see [architecture.md](architecture.md).

---

## Phase 0 — Shell and workspace setup

**Before any ROS processes start**, the script:

1. Sets **`WORKSPACE_DIR`** (defaults to the directory containing `run_llm_nav.sh`, usually `intro/`).
2. Picks the **mission string** from the first argument, or uses the default *“Reach the far side of the obstacle course”*.
3. **`cd`** into the workspace and **`source /opt/ros/humble/setup.bash`** and **`source install/setup.bash`** so ROS and your built `nav2_llm_demo` package are available.
4. **Sources `.env`** (if present) so `LLM_PROVIDER`, `LLM_MODEL`, provider API keys (e.g. `OPENAI_API_KEY`), and optional overrides are exported.
5. Exports **`TURTLEBOT3_MODEL`** (default `burger`) for TurtleBot3 launch files.
6. Validates **`LLM_PROVIDER`**, **`LLM_MODEL`**, the Nav2 map path, `llm_nav_params.yaml`, and `route_graph.json`.
7. Builds the **initial pose** payload string used later for `/initialpose`.

**Logs:** Mostly short `echo` lines from the script; errors here exit before Gazebo starts.

---

## Phase 1 — Gazebo and simulated robot

The script runs:

```bash
ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py &
```

then sleeps a few seconds.

**What happens:** Gazebo starts (`gzserver` and usually `gzclient`), the TurtleBot3 world loads, **`robot_state_publisher`** publishes the robot model, and **`spawn_entity`** places the robot. Plugins publish **`/odom`**, laser **`/scan`**, joint states, and related TF (`odom` → `base_footprint` / `base_link`).

**Typical logs:** `gzserver`, `gzclient`, `robot_state_publisher`, `spawn_entity`, `turtlebot3_diff_drive`, odometry and scan topics coming online. ALSA warnings are common in headless or audio-less setups and are usually harmless.

---

## Phase 2 — Nav2 stack (map, AMCL, planners, optional RViz)

The script runs:

```bash
ros2 launch turtlebot3_navigation2 navigation2.launch.py \
  use_sim_time:=... use_rviz:=... map:=... &
```

then waits **`NAV2_STARTUP_WAIT_SEC`** (default 8 seconds).

**What happens:** Nav2’s composable container brings up lifecycle nodes, including **`map_server`** (loads your `custom_map.yaml` / `.pgm`), **`AMCL`**, and **`lifecycle_manager_navigation`**, which configures and activates **`controller_server`**, **`planner_server`**, local/global costmaps, **`bt_navigator`**, and related nodes. If **`use_rviz`** is true, **RViz** starts and subscribes to map, TF, laser, etc.

**Typical logs:** Long streams from `component_container_isolated` — configuring/activating servers, costmap plugins, bonds to lifecycle managers. RViz may warn about TF or message filters when localization or sim time is still settling.

---

## Phase 3 — Seed AMCL (initial pose)

The script publishes **`/initialpose`** several times (`INITIAL_POSE_RETRIES`, default 5) with a short pause between attempts.

**What happens:** **AMCL** receives an approximate pose in the **`map`** frame so it can publish **`map` → `odom`**. Without a reasonable initial pose, **`map`** may be missing from TF and Nav2 will complain.

**Typical logs:** `Publishing initial pose for AMCL...`, sometimes `Waiting for at least 1 matching subscription(s)...` on `/initialpose` until AMCL’s subscription matches, then `amcl: initialPoseReceived`. If **`use_sim_time`** is true but message stamps do not match simulation clock, you may still see TF or RViz timing warnings until things align.

---

## Phase 4 — LLM navigation node (`llm_nav_node`)

The script runs:

```bash
ros2 run nav2_llm_demo llm_nav_node \
  --ros-args \
  --params-file "$PARAMS_FILE" \
  -p route_graph_path:="$ROUTE_GRAPH_FILE" &
```

then sleeps **`LLM_NODE_WAIT_SEC`** (default 5 seconds).

**What happens:**

1. The node loads **`route_graph.json`** and ROS parameters from **`llm_nav_params.yaml`** (plus `route_graph_path` from the command line).
2. It creates publishers and subscribers (`/navigation_request`, `/navigation_status`, goal pose topic, etc.).
3. It calls **`waitUntilNav2Active()`** so Nav2’s navigation interface is ready before the main loop relies on it.
4. It enters **`rclpy.spin`** and waits for mission strings on **`/navigation_request`**.

**Typical logs:** `Starting LLM decision node (...)` and node logger lines. Import or API errors appear here if Python dependencies or credentials are wrong.

---

## Phase 5 — Status stream and first mission

The script:

1. Starts **`ros2 topic echo /navigation_status`** in the background (duplicates status lines that the node also logs).
2. Publishes a **`std_msgs/String`** once on **`/navigation_request`** with the mission text.

**Mission behavior (inside `llm_nav_node`):**

1. On receiving the mission, the node calls **`plan_route`**, which uses LangChain to query the configured LLM with the **checkpoint graph** and mission text.
2. The returned route is validated against the graph and **blocked edges**; invalid routes are retried up to **`max_decision_attempts`**.
3. For each leg, the node publishes a **goal pose** and uses **`BasicNavigator`** to execute through Nav2; it monitors timeout, stall, and Nav2 results.
4. On segment failure, it may add a **blocked edge** and **replan** (up to **`max_replans`**).
5. Human-readable updates are published on **`/navigation_status`**.

**Typical logs:** `Publishing mission request: ...`, `LLM chose route...`, `Executing segment ...`, `Ignoring invalid route from LLM...`, or `Mission failed: ...` / mission complete messages, interleaved with Nav2 and AMCL if the stack resets or localization degrades.

---

## Phase 6 — Run until Ctrl+C

The script runs **`wait`** so the foreground shell stays alive while background jobs (Gazebo, Nav2, `llm_nav_node`, topic echo) keep running.

**Stopping:** **Ctrl+C** triggers the script’s **`cleanup`** trap, which stops the background PIDs. You may see rclpy shutdown messages or a **double shutdown** traceback; that is a common teardown artifact when the ROS context is shut down from multiple places.

---

## Quick reference

| Phase | Main components | Purpose |
|------|-----------------|--------|
| 0 | Bash, `.env`, workspace | Configure environment and validate inputs |
| 1 | `turtlebot3_gazebo` | Simulation and robot sensors |
| 2 | `turtlebot3_navigation2` | Map, localization, planning, control |
| 3 | `/initialpose` pubs | Give AMCL an initial estimate |
| 4 | `llm_nav_node` | Graph + LLM mission controller |
| 5 | `/navigation_status` echo + `/navigation_request` pub | Visibility and default mission |
| 6 | `wait` + cleanup | Keep stack running until user stops |

---

## Related files

- `intro/run_llm_nav.sh` — orchestration script
- `intro/src/nav2_llm_demo/config/llm_nav_params.yaml` — node parameters
- `intro/src/nav2_llm_demo/config/route_graph.json` — LLM-visible checkpoint graph
- `intro/src/nav2_llm_demo/nav2_llm_demo/llm_nav_node.py` — ROS node implementation
- `intro/src/nav2_llm_demo/nav2_llm_demo/llm/llm_routing.py` — LLM call and route validation
