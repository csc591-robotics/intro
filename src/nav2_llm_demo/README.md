# `nav2_llm_demo` — LLM-driven Nav2 missions

A high-level navigation layer on top of Nav2. The LLM decides **which
checkpoint to head to next**; Nav2 does **all** the actual driving,
costmap reasoning, obstacle avoidance, and recovery.

## What this package now does

A single launch file (`llm_nav.launch.py`, wrapped by
`scripts/run_llm_nav.sh`) brings up everything end-to-end so that
**Gazebo and RViz both show the robot in the same place from frame
zero**:

1. Reads the entry for the chosen `map_name` from the shared
   `src/custom_map_builder/maps/map_poses.yaml` (the single source of
   truth — same file `custom_map_builder` writes to).
2. Reads the per-map `.world_map.yaml` *sidecar* it points to (produced
   by `world_to_map.rasterize_world`) for the `.world` file path,
   `world_to_map_offset`, and Nav2 `map.yaml`.
3. Launches **Gazebo** with that `.world`.
4. Spawns the **TurtleBot3** at the `source` pose recorded in
   `map_poses.yaml` (Gazebo world frame).
5. Publishes a static `map → odom` TF whose translation equals
   `source + world_to_map_offset` so the robot's pose in the `map`
   frame matches its position in Gazebo to the millimetre.
6. Brings up `map_server` + the Nav2 **navigation** stack (planner,
   controller, bt_navigator, behavior, smoother, waypoint_follower).
   AMCL is intentionally **not** used: Gazebo's diff-drive plugin gives
   us perfect odometry and our static TF gives us perfect localization,
   so AMCL would only fight us.
7. Opens **RViz** on the rasterized PGM with the camera focused on the
   robot.
8. Starts `llm_nav_node` parameterised with the same
   `map_poses_path` + `map_name` so the LLM picks goals from the
   `route_graph` block of that entry.

For hand-crafted maps with no sidecar (`sidecar: null` in
`map_poses.yaml`), step 3–5 are skipped and the launch falls back to
plain `nav2_bringup` (full bringup with AMCL). You must pass
`map_yaml_fallback:=…` so Nav2 has a map to load.

## Quick start

```bash
cd /workspace/intro
colcon build --packages-select nav2_llm_demo world_to_map
source install/setup.bash

# Most common case — a world_to_map-generated map:
bash src/nav2_llm_demo/scripts/run_llm_nav.sh warehouse

# In a second terminal, send a mission:
ros2 topic pub --once /navigation_request std_msgs/String \
  "data: 'goal'"
```

Available map names (taken from `map_poses.yaml`): `warehouse`,
`workshop_example`, `test_zone`, `diamond_blocked` (hand-crafted, needs
`MAP_YAML_FALLBACK`).

### Hand-crafted map (no Gazebo)

```bash
MAP_YAML_FALLBACK=src/custom_map_builder/maps/diamond_blocked.yaml \
  bash src/nav2_llm_demo/scripts/run_llm_nav.sh diamond_blocked
```

## How `map_poses.yaml` ties it all together

The file `src/custom_map_builder/maps/map_poses.yaml` is shared with
`custom_map_builder` and contains, per map:

```yaml
maps:
  warehouse.pgm:
    sidecar: src/world_to_map/maps/warehouse.world_map.yaml   # used to find .world + offset
    source:                                                   # robot spawn pose (Gazebo world frame)
      position: {x: 0.0, y: 0.0, z: 0.0}
      orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}
      yaw_rad: 0.0
    destination: {...}                                        # (informational)
    route_graph:                                              # consumed by llm_nav_node
      start_checkpoint: start
      checkpoints:
        start: {x: 0.0,  y: 0.0,  yaw: 0.0, description: "..."}
        shelf: {x: 1.5,  y: 3.4,  yaw: 0.0, description: "..."}
      edges:
        - {from: start, to: shelf}
      goal_aliases:
        shelf:   [shelf]
        deliver: [shelf]
```

All `(x, y)` are in the **Gazebo world frame**. The launch file applies
the sidecar offset before sending goals to Nav2.

### Workflow for filling in coordinates

1. Run `custom_map_builder` to *just click points* (no Gazebo, no LLM):
   ```bash
   LAUNCH_GAZEBO=false bash src/custom_map_builder/scripts/run_map_builder.sh warehouse
   ```
   In RViz press **P** (Publish Point) and click. Each click prints:
   ```
   frame="map"  MAP  x=9.42  y=13.10  |  GAZEBO WORLD  x=0.73  y=1.34
   ```
2. Copy the **GAZEBO WORLD** values into the matching `route_graph`
   entry of `map_poses.yaml`.
3. `nav2_llm_demo` will apply the sidecar offset automatically when
   sending goals to Nav2.

## End-to-end LLM mission flow

1. A user publishes a text mission request on `/navigation_request`.
2. `llm_nav_node` loads the current graph state from `map_poses.yaml`.
3. The LLM chooses a legal route through allowed edges.
4. The node validates that route.
5. The node sends the next checkpoint pose to Nav2 (after applying the
   `world → map` offset).
6. Nav2 uses costmaps + sensors to drive there.
7. If a segment fails, that edge is marked blocked and the node replans.

## What the LLM sees vs what Nav2 sees

The LLM does **not** receive camera images, lidar scans, point clouds,
or raw TF. It receives a structured graph context:

- the natural-language mission request
- the robot's current checkpoint
- goal aliases
- checkpoint descriptions
- allowed edges
- blocked edges
- the last failure reason

So the split is:

- **LLM**: "Which checkpoint route should we try?"
- **Nav2**: "How do I physically get to the next pose safely?"

## Topics

- `/navigation_request` (`std_msgs/String`) — incoming mission text
- `/navigation_status`  (`std_msgs/String`) — human-readable status
- `/active_goal_pose`   (`geometry_msgs/PoseStamped`) — the current goal
  Nav2 is chasing (already in `map` frame after offset is applied)

## Files

- `launch/llm_nav.launch.py` — orchestrator
- `scripts/run_llm_nav.sh` — convenience wrapper
- `nav2_llm_demo/llm_nav_node.py` — the mission controller
- `nav2_llm_demo/llm/llm_routing.py` — graph loader + LLM helpers
- `config/llm_nav_params.yaml` — node parameters
