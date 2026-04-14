# `llm_nav_node` Overview

This package is a high-level navigation layer on top of Nav2.

It does not do localization, obstacle detection, or low-level path following by itself. Its job is:

1. Read a simple route graph from `config/route_graph.json`.
2. Send the current mission context to the LLM.
3. Get back a checkpoint route such as `start -> north_staging -> north_pass -> goal_zone`.
4. Convert each checkpoint into a `PoseStamped` goal in the `map` frame.
5. Hand those goals to Nav2 one segment at a time.

## How the map works

There are two different "map" concepts in this system:

- `route_graph.json`: a simple directed graph used by the LLM for high-level decisions.
- Nav2 map / costmaps: the real navigation map and obstacle layers used by Nav2 to physically move the robot.

### 1. The route graph

The JSON graph defines:

- `start_checkpoint`: where the robot starts in graph terms
- `checkpoints`: named poses with `x`, `y`, `yaw`, and a text `description`
- `edges`: allowed directed transitions between checkpoints
- `goal_aliases`: user-friendly goal names that map to one or more final checkpoints

Example:

- `start -> south_staging`
- `south_staging -> south_pass`
- `south_pass -> goal_zone`

This means the LLM is only allowed to choose among the listed nodes and edges. It cannot invent a new route or a new coordinate.

### 2. The real Nav2 map

Nav2 still needs its own normal navigation stack:

- `map_server`
- `amcl` or another localization source
- global and local costmaps
- planner/controller servers
- TF frames like `map`, `odom`, and `base_link`

That is the map Nav2 uses to actually drive the robot around obstacles.

## What data the LLM receives

The LLM does not receive live camera images, lidar scans, point clouds, or raw TF.

It receives structured graph-level context built in `llm_nav_node.py`, including:

- the natural-language mission request
- the robot's current checkpoint
- goal aliases
- checkpoint descriptions
- allowed edges
- blocked edges
- the last failure reason

In other words, the LLM sees a simplified symbolic routing problem, not raw sensor perception.

## What data Nav2 receives

Nav2 receives the actual pose goals and uses robot/navigation data to execute them.

Nav2 typically depends on:

- the static map
- localization output
- TF transforms
- odometry
- lidar or depth obstacles
- costmap updates

So the split is:

- LLM: "Which checkpoint route should we try?"
- Nav2: "How do I physically get to the next pose safely?"

## Are obstacles in `route_graph.json`?

Not directly.

The graph does not contain obstacle objects. Obstacles are handled by Nav2 using the real map and live sensors.

The graph can still reflect obstacle-aware design indirectly by only including safe corridor choices. For example, having `north_pass` and `south_pass` but no direct middle edge effectively tells the LLM to route around an obstacle region.

## What are blocked edges?

If Nav2 fails on a segment, the node stores that segment as a blocked edge for the current mission.

Example:

- Nav2 fails on `south_staging -> south_pass`
- that edge is added to `blocked_edges`
- the next LLM call is told not to use it again

That lets the system replan around a failed branch without giving the LLM raw sensor data.

## End-to-end flow

1. A user sends a text mission request on `/navigation_request`.
2. `llm_nav_node` loads the current graph state.
3. The LLM chooses a legal route through allowed edges.
4. The node validates that route.
5. The node sends the next checkpoint pose to Nav2.
6. Nav2 uses localization, costmaps, and sensors to drive there.
7. If a segment fails, that edge is marked blocked and the node replans.

## Important limitation

This package depends on Nav2 already being properly started and localized.

If `map`, `odom`, and `base_link` are not available, or AMCL has not been initialized, the LLM node cannot navigate because it only provides high-level decisions and goal poses. Nav2 is the part that actually moves the robot.

## Step 2 tools and agent loop

Step 2 adds:

- a ROS-backed tool backend that reads pose, lidar, camera, and annotated-map state
- a local stdio MCP server exposing the required robot tools
- a LangGraph mission loop that senses state, decides the next move, navigates, confirms the result, and repeats

The MCP server is intended to be started directly by the LLM client process over stdio rather than launched as a normal ROS node.

Required tools:

- `navigate_to_checkpoint`
- `navigate_to_coordinates`
- `report_blockade`
- `get_robot_position`
- `get_camera_snapshot`
- `get_annotated_map`
- `get_lidar_summary`

Assumptions called out in the implementation:

- Step 1 will eventually publish or save annotated-map artifacts
- camera input may be absent in some worlds
- named checkpoints remain backed by `route_graph.json` in v1
