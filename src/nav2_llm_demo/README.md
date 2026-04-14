# `llm_agent_node` Overview

This package provides a **vision-based LLM agent** that directly controls a TurtleBot3 robot to navigate custom occupancy-grid maps.

Unlike a traditional Nav2 waypoint approach, the LLM **sees** the map and **controls** the robot through tool calling (move forward, rotate, check position).

## How It Works

1. A custom map (PGM + YAML) is loaded into `map_server` and displayed in RViz.
2. The robot spawns in Gazebo at the **source pose** defined in `maps/nav_config.yaml`.
3. A LangGraph ReAct agent receives:
   - An annotated top-down map image (robot = red arrow, destination = green circle)
   - Tools to move forward, rotate, check pose, and check goal arrival
4. The agent iteratively observes the map and issues movement commands via `cmd_vel`.
5. Navigation continues until the robot reaches the destination or the step limit is exhausted.

## Map Configuration

Maps live in `maps/` and are configured in `maps/nav_config.yaml`:

```yaml
frame_id: map

maps:
  diamond_blocked:
    map_yaml: diamond_blocked.yaml
    source:
      position: {x: 13.1001, y: 11.3377, z: 0.0}
      orientation: {x: 0.0, y: 0.0, z: 0.991511, w: 0.130025}
    destination:
      position: {x: 10.2383, y: 19.1664, z: 0.0}
      orientation: {x: 0.0, y: 0.0, z: 0.723957, w: 0.689845}
```

To add a new map:

1. Build/test the map in `custom_map_builder`.
2. Copy the `.pgm` and `.yaml` files into `nav2_llm_demo/maps/`.
3. Add an entry in `nav_config.yaml` with source and destination poses.
4. Rebuild: `colcon build --packages-select nav2_llm_demo`

## Running

```bash
bash ./run_llm_nav.sh MAP_NAME
```

Example:

```bash
bash ./run_llm_nav.sh diamond_blocked
```

The script:
- Parses `nav_config.yaml` for the given map name
- Starts Gazebo with an empty world and spawns the robot at the source pose
- Starts `map_server` with the selected map
- Launches the LLM agent node
- Streams `/navigation_status` updates

## Agent Tools

The LLM has these tools available:

| Tool | Description |
|------|-------------|
| `get_map_view()` | Captures an annotated top-down map image |
| `move_forward(distance_meters)` | Drives forward/backward via `cmd_vel` |
| `rotate(angle_degrees)` | Rotates in place (positive = left/CCW) |
| `get_robot_pose()` | Returns current x, y, yaw |
| `check_goal_reached()` | Checks proximity to destination |

## Requirements

- **Vision-capable LLM**: GPT-4o, Claude 3.5 Sonnet, Gemini Pro, etc.
- Set `LLM_PROVIDER` and `LLM_MODEL` in `.env`
- Python packages: `langchain`, `langgraph`, `Pillow`, `numpy`

## Architecture

```
run_llm_nav.sh
├── Gazebo (empty world + TurtleBot3 at source pose)
├── map_server (loads PGM/YAML)
├── static TF (map -> odom)
└── llm_agent_node
    ├── subscribes: /odom
    ├── publishes: /cmd_vel, /navigation_status
    ├── TF listener: map -> base_link
    └── LangGraph ReAct agent
        ├── get_map_view() -> annotated PNG -> vision LLM
        ├── move_forward() -> cmd_vel + odom monitoring
        ├── rotate() -> cmd_vel + odom monitoring
        ├── get_robot_pose() -> TF lookup
        └── check_goal_reached() -> distance check
```

## Important Notes

- The Gazebo world is **empty** (no physical walls). The LLM navigates based on the 2D occupancy grid image. This means the robot won't physically collide with map obstacles in simulation.
- The agent bypasses Nav2 entirely and controls the robot directly via velocity commands.
- Movement is monitored via odometry to ensure the requested distance/rotation is achieved.
- The agent has a configurable step limit (`max_agent_steps`, default 50) to prevent infinite loops.
