# LLM Flow Detailed Summary

This document explains the five navigation-agent flows implemented under
`src/nav2_llm_demo/nav2_llm_demo/llm`. The active flow is selected with the
`LLM_FLOW` environment variable. If `LLM_FLOW` is not set, the package defaults
to flow 1.

All flows expose the same basic public surface through `build_agent()`:

- `initialize(source_x, source_y, dest_x, dest_y)`
- `step()`
- `goal_reached_in_last_step`
- `terminated` where implemented
- `run_dir`

That shared surface lets the ROS node swap flows without needing flow-specific
conditionals. Internally, though, the flows are very different. Flow 1 is a
hand-written tool loop, flows 2 and 3 use LangGraph's ReAct agent, flow 4 uses
a fixed gather-decide-execute-check loop, and flow 5 plans an A* path before
asking the LLM to follow it.

## Shared Pieces

### Controller Registry

The shared controller code lives in `llm/controller.py`. The ROS node registers
the active `RobotController` with `set_controller()`, and the flow tools call
`get_controller()` whenever they need to:

- read the robot pose with `get_pose()`
- drive with `move_forward(distance_m)`
- turn with `rotate(angle_deg)`
- access the latest LiDAR scan with `get_latest_scan()` when the flow uses it
- read map, source, and destination metadata

The controller protocol is intentionally small. The LLM modules do not need to
import ROS message types or know about the ROS node's internal state. They only
call this stable interface.

### Per-Flow LLM Configuration

Each flow resolves its model through `resolve_llm_config(flow)`. The lookup
order is:

1. `FLOW<N>_LLM_PROVIDER` and `FLOW<N>_LLM_MODEL`
2. `LLM_PROVIDER` and `LLM_MODEL`

For example, flow 3 first checks `FLOW3_LLM_PROVIDER` and
`FLOW3_LLM_MODEL`, then falls back to the global settings. This makes it
possible to run different flows against different model/provider combinations
without changing code.

### Run Directories And Logging

Each flow writes artifacts under:

```text
<WORKSPACE_DIR>/llm_agent_runs/flow_<N>/<timestamp>/
```

`WORKSPACE_DIR` defaults to `/workspace`, which maps to the project folder in
the container setup. Each LLM request gets its own folder:

```text
llm_controls_call_001/
  request.json
  response.json
  image_sent.png   # only when an image was included in the request
```

Flow 1 writes these artifacts directly. Flows 2 through 5 reuse
`flow_2.logging.PerCallLogger`, a LangChain callback handler that records the
messages sent to the chat model and the model's tool-call response.

### Map Rendering

The vision-based flows use annotated top-down map images. The shared renderer
draws:

- blue marker for the source/start position
- red dot for the robot's current position
- red arrow for the robot's heading
- green plus/circle for the destination
- white pixels as free space
- black pixels as obstacles/walls
- gray pixels as unknown space

The standard crop is a local 18 meter radius rendered to a 512 by 512 PNG. The
image is base64 encoded and sent as a multimodal message block.

### History Compaction

Flows 2 through 5 can run for many turns and include large image messages, so
they use two message utilities:

- `compact_history()` keeps the original goal message and a recent window of
  tool/action rounds, dropping the middle of the conversation.
- `prune_old_images()` replaces stale inline image blocks with a text
  placeholder, keeping only the most recent image verbatim.

This keeps requests small enough for provider rate limits while preserving the
latest map and recent action context.

## Flow 1: Hand-Rolled Vision Tool Loop

### Purpose

Flow 1 is the original custom LLM control loop. It gives the model full tool
freedom and manually injects map images into the conversation after the model
calls `get_map_view()`.

This flow is closest to a simple "LLM agent controls robot directly" design:
the model sees the map, decides whether to rotate, move, ask for pose, refresh
the map, or check the goal, and the Python loop executes whatever tool call the
model emits.

### Initialization

`Flow 1` is implemented by `flow_1.agent.VisionNavigationAgent`. During
construction it:

- resolves the provider/model for flow 1
- creates a chat model with temperature `0.1`
- binds raw OpenAI-style tool schemas with `bind_tools()`
- initializes an empty message history
- delays run-directory creation until logging is needed

`initialize()` creates a conversation with:

- a `SystemMessage` containing the full navigation rules from
  `flow_1.prompt.SYSTEM_PROMPT`
- a `HumanMessage` containing the source and destination coordinates

The first user instruction tells the model to begin by calling
`get_map_view()`.

### Tools

Flow 1 exposes five tools:

- `move_forward(distance_meters)` drives forward for positive values or
  backward for negative values.
- `rotate(angle_degrees)` rotates in place, where positive means left/CCW and
  negative means right/CW.
- `get_map_view()` renders an annotated map image.
- `get_robot_pose()` returns exact `x`, `y`, and yaw in degrees as JSON.
- `check_goal_reached()` compares the current robot position to the
  destination and returns `GOAL REACHED` when within `0.3 m`.

The prompt asks the model to use small movement steps, usually `0.3` to
`1.0 m`, and to call `get_map_view()` first and after every few moves.

### One Step

Each `step()` call performs exactly one LLM round trip:

1. Increment the step counter.
2. Log the current message list to `request.json`.
3. Invoke the model with the current messages and bound tools.
4. Log the model response to `response.json`.
5. Append the model response to the conversation.
6. Execute every tool call emitted by the model.
7. Append a `ToolMessage` with each tool's text result.
8. If a tool result included a map image, append an extra multimodal
   `HumanMessage` containing the image and the repeated map legend.

The important detail is step 8. In flow 1, `get_map_view()` does not return the
image directly inside the tool result. Instead, it returns text saying a map was
captured, and then the agent injects a separate `HumanMessage` containing:

- the `MAP_IMAGE_CONTEXT` legend
- the base64 PNG as an `image_url`

That means the model receives the image as if the user had sent another message
after the tool ran.

### Prompt Behavior

The flow 1 system prompt strongly emphasizes:

- black pixels are forbidden obstacles
- white pixels are safe driveable space
- gray is unknown and should be treated as unsafe
- the red arrow is the robot heading
- the map image uses standard orientation, with `+X` right and `+Y` up
- every reply must be a tool call

The repeated `MAP_IMAGE_CONTEXT` is attached to every map image because long
chat histories can dilute the original system prompt. This keeps the legend and
safety rules next to the image the model is currently inspecting.

### Termination

Flow 1 does not own a terminal graph state. Its `terminated` property always
returns `False`. The ROS-side loop keeps calling `step()` until external logic
stops or until `goal_reached_in_last_step` notices a recent message containing
`GOAL REACHED`.

### Strengths And Weaknesses

Flow 1 is easy to understand and useful for debugging because the loop is
explicit Python. However, it depends heavily on the model to choose sensible
tool order. The model can choose to keep looking, chain actions, or forget to
check progress unless the prompt prevents it. It also grows history quickly
because map images are appended as separate human messages.

## Flow 2: LangGraph ReAct With Multimodal Tool Results

### Purpose

Flow 2 moves from a hand-written loop to LangGraph's prebuilt
`create_react_agent()`. It keeps the same high-level tool set as flow 1, but
changes how images are delivered and how the agent loop is executed.

The goal of flow 2 is to let LangGraph handle the ReAct loop:

```text
LLM -> tool calls -> ToolNode -> LLM -> tool calls -> ToolNode -> ...
```

The graph continues until the model stops emitting tool calls or the recursion
limit is reached.

### Initialization

`Flow2Agent` creates:

- a chat model at temperature `0.1`
- a LangGraph ReAct graph with `create_react_agent()`
- a recursion limit, defaulting to `50`
- a per-call logger

`initialize()` creates only the initial goal `HumanMessage`. The system prompt
is passed to `create_react_agent()`, so the caller does not manually prepend a
`SystemMessage` in the message state.

The initial instruction again tells the model to begin with `get_map_view()`.

### Tools

Flow 2 tools are implemented with LangChain's `@tool` decorator:

- `move_forward(distance_meters)`
- `rotate(angle_degrees)`
- `get_map_view()`
- `get_robot_pose()`
- `check_goal_reached()`

The major change is `get_map_view()`. It uses
`response_format="content_and_artifact"` and returns:

- content: a list with text plus an `image_url` block
- artifact: metadata containing the raw base64 image for logging

LangGraph wraps the content list as a multimodal `ToolMessage`. No extra
`HumanMessage` is injected. The map image is delivered as the direct output of
the `get_map_view()` tool.

### One Step

Flow 2 treats one `step()` call as one full graph run. Inside `step()`:

1. If the graph already completed, return the cached final summary.
2. Create the run directory and callback logger if needed.
3. Compact the current history to keep the first goal message and the last
   four tool/action rounds.
4. Prune old images, keeping only the latest inline image.
5. Invoke the LangGraph ReAct graph with the compacted/pruned messages.
6. Let LangGraph alternate model calls and tool calls internally.
7. Store the resulting messages.
8. Summarize the final assistant/tool behavior.
9. Mark the flow as completed.

Because `create_react_agent()` runs its internal loop until completion, flow 2
does not run one physical robot action per `step()`. A single `step()` can
contain many LLM calls and many tool calls.

### Prompt Behavior

The flow 2 prompt is stricter than flow 1 about refreshing the map. It says:

- every reply must be exactly one tool call
- `get_map_view()` should be called first
- after every `move_forward` or `rotate`, the next action must be
  `get_map_view()`
- the model should not chain movement commands without fresh visual feedback

The map tool text repeats the legend directly beside the image and explicitly
warns that after any movement the next call must be `get_map_view()`.

### Termination

Flow 2 sets `_completed = True` after the graph returns, hits recursion limit,
or raises an error. `terminated` then returns `True`.

`goal_reached_in_last_step` searches recent messages for a `GOAL REACHED`
marker. The goal threshold is `0.3 m`.

### Strengths And Weaknesses

Flow 2 is cleaner than flow 1 because LangGraph owns the ReAct loop and map
images are proper multimodal tool results. It also logs requests and responses
through a reusable callback.

The main tradeoff is that the model still has ReAct freedom. It can spend many
turns looking, checking pose, rotating, or otherwise deciding what to do. The
prompt tries to force a look-act-look cycle, but the graph topology does not
guarantee that behavior.

## Flow 3: ReAct With Map Plus LiDAR

### Purpose

Flow 3 builds on flow 2 by adding live LiDAR awareness. It still uses
LangGraph's `create_react_agent()`, but gives the model more situational tools:

- raw sector LiDAR summaries
- a combined map-plus-LiDAR `get_situation()` tool
- an optional smaller interpreter LLM that converts raw LiDAR numbers into a
  plain-English safety read

The intent is to help the model cross-check the static raster map against what
the robot sees physically in front of it.

### Initialization

`Flow3Agent` is structurally almost identical to `Flow2Agent`, with these
changes:

- it resolves flow 3 model settings
- it uses the flow 3 system prompt
- it binds the flow 3 tool list
- it writes logs under `flow_3`
- its default recursion limit is `50`

The recursion limit keeps the ReAct loop from running effectively unlimited
inside one ROS-side agent step.

The initial human message tells the model to call `get_map_view()` and
`get_lidar_summary()` before deciding the first action. The prompt itself later
recommends `get_situation()` as the preferred combined tool.

### LiDAR Sector Summary

`flow_3.lidar.summarize_scan()` converts the latest scan into six sectors:

- `front`: -30 to +30 degrees
- `front_left`: +30 to +90 degrees
- `back_left`: +90 to +150 degrees
- `back`: +150 to +180 and -180 to -150 degrees
- `back_right`: -150 to -90 degrees
- `front_right`: -90 to -30 degrees

Each sector reports the closest valid range and a status:

- `BLOCKED` when distance is below `0.40 m`
- `CAUTION` when distance is below `1.00 m`
- `CLEAR` otherwise
- `CLEAR (no return)` when no finite obstacle return exists

The summary ends with a headline such as "forward is CLEAR" or "avoid front".
The front sector is treated as the most important number because it determines
whether a positive `move_forward()` is immediately safe.

### LiDAR Interpreter

`flow_3.interpreter.interpret_lidar()` optionally calls a second LLM to turn
the raw sector table into a short navigation safety read. Its default model is
configured with:

- `FLOW3_INTERPRETER_LLM_PROVIDER`, falling back to `FLOW3_LLM_PROVIDER` and
  then `LLM_PROVIDER`
- `FLOW3_INTERPRETER_LLM_MODEL`, defaulting to `claude-haiku-4-5`

The interpreter is instructed to:

- make the first sentence about the front sector
- explicitly name other blocked sectors
- optionally recommend a safe immediate action
- stay terse and avoid inventing values

If the interpreter call fails, the flow falls back to sending the raw sector
table to the main model.

### Tools

Flow 3 exposes:

- `move_forward(distance_meters)`
- `rotate(angle_degrees)`
- `get_map_view()`
- `get_lidar_summary()`
- `get_situation()`
- `get_robot_pose()`
- `check_goal_reached()`

`get_situation()` is the preferred tool. It returns one multimodal tool result
containing:

- a text header explaining the map legend
- the LiDAR analyst read
- a fresh annotated map image
- hidden artifact metadata with the image, raw LiDAR summary, and interpreted
  LiDAR text

This saves an LLM round trip compared with calling `get_map_view()` and
`get_lidar_summary()` separately.

### One Step

Flow 3's `step()` is the same shape as flow 2:

1. Return the cached summary if already completed.
2. Compact history to the first message plus the last four action rounds.
3. Inject the flow 3 reminder text when history was pruned.
4. Prune stale images, keeping only the latest image.
5. Invoke the LangGraph ReAct graph.
6. Let the graph run LLM/tool turns until it stops or hits the recursion limit.
7. Store final messages, summarize, and mark completed.

Because it is still ReAct, the LLM can choose any available tool at each turn.
The prompt strongly prefers `get_situation()` before every movement decision,
but the graph does not hard-code that sequence.

### Prompt Behavior

Flow 3 adds several absolute rules:

- black pixels in the PGM map are forbidden and cannot be overridden by LiDAR
- if the red dot is on or touching black, the model must escape first
- rotation direction is relative to the red arrow
- LiDAR is only short-range safety reinforcement
- if front LiDAR is blocked, the model should rotate before moving forward

The prompt explicitly says LiDAR cannot grant permission to enter a black map
region. The map remains the authority for where the robot is allowed to go,
while LiDAR warns about immediate obstacles the map might miss.

### Termination

Flow 3 terminates when the LangGraph run returns, hits recursion limit, or
throws an error. `goal_reached_in_last_step` checks recent messages for
`GOAL REACHED`. The code-level goal threshold is `0.3 m`.

### Strengths And Weaknesses

Flow 3 improves physical safety by giving the model live range data. It can
avoid moving forward when something is close even if the map looks clear.

The tradeoff is complexity. There is more prompt text, more tools, and
potentially more LLM calls because the LiDAR interpreter can run inside
`get_situation()`. The main model also still has full ReAct freedom, so it can
spend many cycles observing or choose suboptimal tool order.

## Flow 4: Fixed Gather-Decide-Execute-Check Cycle

### Purpose

Flow 4 removes ReAct freedom. Instead of letting the model choose perception
tools, Python gathers the situation every cycle, and the LLM is only allowed to
choose one action:

- `move_forward`
- `rotate`

This is the first flow where the graph topology enforces "always act after
observing." The LLM cannot decide to call `get_map_view()`, call
`get_lidar_summary()`, or send free text. It receives the latest situation and
must pick a movement tool.

### Initialization

`Flow4Agent`:

- resolves flow 4 model settings
- creates a chat model with temperature `0.0`
- binds only `move_forward` and `rotate`
- tries to use `tool_choice="any"` so the provider forces a tool call
- initializes cycle state and destination coordinates

`initialize()` stores the destination and creates a preserved preamble
`HumanMessage` saying that each turn will provide a fresh map-plus-LiDAR
situation and that the model must answer with one movement tool call.

The actual code-level goal threshold is `0.3 m`.

### Situation Gathering

Flow 4 uses `flow_4.situation.build_situation_message()`. This function runs in
Python before the LLM call. It:

1. Reads the robot pose.
2. Renders an annotated map image.
3. Reads the latest LiDAR scan if available.
4. Converts the scan to sectors with `summarize_scan()`.
5. Converts the sector summary to plain English with `interpret_lidar()`.
6. Creates a multimodal `HumanMessage` containing text plus the map image.

The LLM receives the result as ordinary conversation context, not as a tool
result. The LLM does not know how to request another observation; it always gets
a fresh one at the start of the next cycle.

### One Cycle

Each `step()` runs exactly one control cycle:

1. Increment the cycle counter.
2. Gather a fresh situation in Python.
3. Append the situation message to history.
4. Compact the history and prune old images.
5. Prepend `DECIDE_SYSTEM_PROMPT`.
6. Invoke the LLM with only `move_forward` and `rotate` bound.
7. If no tool call is returned, stop with an error summary.
8. Execute the selected movement tool through the controller.
9. Append a `ToolMessage` with the execution result.
10. Read the current pose.
11. Check distance to the destination.
12. If within `0.3 m`, mark goal reached and terminate.
13. Otherwise, return a cycle summary and wait for the next `step()`.

Unlike flows 2 and 3, one `step()` here corresponds to one action cycle, not an
entire autonomous ReAct episode.

### Prompt Behavior

The decision prompt is short and action-oriented. It tells the model:

- black map pixels are forbidden
- if the red dot is on black, the only valid action is `rotate(180)`
- rotation is relative to the red arrow
- if front LiDAR is blocked, rotate first
- if front is clear and the arrow points toward the destination, move forward
- otherwise rotate decisively toward the destination

The policy explicitly discourages tiny repeated corrective rotations. If the
model rotated last cycle and front is now clear, the prompt says the next move
should usually be forward.

### Termination

Flow 4 sets `_completed` when:

- the goal is reached
- the LLM returns no tool call
- an LLM/tool exception occurs

`goal_reached_in_last_step` returns the internal `_goal_marker`, which is set
only when the distance check passes.

### Strengths And Weaknesses

Flow 4 is more controlled than ReAct. Every cycle has fresh perception, one
decision, one action, and a deterministic goal check. The model cannot waste
turns looking without acting.

The tradeoff is that it still asks the LLM to infer a navigation action from
the image. The model must judge whether the arrow is aligned, where the green
destination is, and how to steer around walls. It is less open-ended than flow
3, but still visually reasons about local routing every cycle.

## Flow 5: A* Planner Plus LLM Path Follower

### Purpose

Flow 5 is the most deterministic flow. It plans a path once with A* on the PGM
map, draws that path as a magenta line, computes the target waypoint and
bearing each cycle, and asks the LLM to follow the suggested action.

The LLM is no longer responsible for global route planning. Its job is mostly
to convert the suggested `rotate(...)` or `move_forward(...)` action into the
correct tool call.

### Initialization

`Flow5Agent`:

- resolves flow 5 model settings
- creates a temperature `0.0` chat model
- binds only `move_forward` and `rotate`
- uses `tool_choice="any"` when supported
- initializes planned path state and target waypoint index

During `initialize()`, flow 5 immediately calls `plan_astar_path()` with the
source and destination coordinates. If planning raises an exception or returns
no waypoints, the agent marks itself completed and stores a failure summary.

If planning succeeds, the initial message says:

- where the robot starts
- where it must go
- how many waypoints A* produced
- that the path will be drawn as a magenta line
- that the LLM should default to the suggested action each turn

### A* Planning Pipeline

The planner in `flow_5.path_planner` is pure Python and depends on NumPy,
Pillow, and `heapq`. It does not import ROS or LangChain.

The planning pipeline is:

1. Load the map YAML metadata and PGM image.
2. Apply `negate` if the map metadata requests it.
3. Threshold the image into a binary obstacle grid where obstacles are pixels
   at or below `occ_threshold`, default `100`.
4. Inflate obstacles by `inflation_px`, default `3`, so the path stays a few
   pixels away from walls.
5. Convert source and destination world coordinates into map pixels.
6. Snap source and destination to the nearest free pixel if either lies inside
   the inflated obstacle grid.
7. Run 8-connected A* with Euclidean heuristic.
8. Reconstruct the pixel path from the A* parent map.
9. Downsample the path into waypoints spaced about `0.6 m` apart.
10. Convert waypoint pixels back into map-frame meter coordinates.

If no free snapped endpoint or no A* path exists, the planner returns an empty
list and the agent stops.

### Path Rendering

Flow 5 uses `flow_5.renderer.render_with_path()` instead of the shared basic
map renderer. It draws:

- magenta polyline for the planned A* path
- small magenta dots for unreached waypoints
- cyan crosshair for the current target waypoint
- red robot dot and heading arrow
- blue source marker
- green final destination marker

The crop is expanded to include the robot, current target, final destination,
and source when available. This makes the follower image focus on the local
tracking problem while still showing the goal context.

### Per-Cycle Situation

`flow_5.situation.build_situation_message()` computes deterministic guidance
before the LLM sees the image. It:

1. Reads the robot pose.
2. Selects the current target waypoint.
3. Computes the vector from robot to target.
4. Computes distance to target.
5. Computes world-frame bearing to target.
6. Computes `bearing-from-heading` by subtracting the robot yaw.
7. Normalizes that angle to `[-180, +180]`.
8. Produces a suggested action:
   - `rotate(+/-degrees)` if absolute bearing error is above `15 degrees`
   - `move_forward(distance)` otherwise
9. Caps forward steps at `0.6 m`, with a minimum of `0.15 m`.
10. Renders the path image.
11. Sends both the text guidance and image to the LLM.

The text block includes exact pose, current target coordinates, target
distance, bearing offset, final goal distance, remaining waypoints, and the
suggested action.

### One Cycle

Each `step()` does one follower cycle:

1. Return the cached summary if already completed.
2. Increment the cycle counter.
3. Auto-advance the target waypoint while the robot is within `0.5 m` of it.
4. Check whether the final destination is already within `0.3 m`.
5. Build the path-follower situation message.
6. Compact history and prune old images.
7. Prepend `FOLLOWER_SYSTEM_PROMPT`.
8. Invoke the LLM with only `move_forward` and `rotate` bound.
9. Execute the tool call through the controller.
10. Append the tool result.
11. Auto-advance the target waypoint again.
12. Check final goal distance again.
13. Mark goal reached if within `0.3 m`; otherwise return a progress summary.

### Prompt Behavior

The follower prompt tells the model that:

- the magenta line was planned by A*
- the path is guaranteed to stay in free white space
- the LLM does not need to reason about black pixels
- the cyan crosshair is the active target
- the text block already computes the required bearing and suggested action
- the default behavior is to copy the suggested action verbatim
- the model should only override if the image clearly contradicts the
  suggestion

This prompt intentionally narrows the LLM's role. It is a command follower, not
a planner.

### Termination

Flow 5 terminates when:

- A* planning fails
- A* returns no path
- the goal is reached
- the LLM returns no tool call
- an exception occurs during the cycle

The code-level final goal threshold is `0.3 m`. Waypoint advancement uses the
looser `0.5 m` threshold so the robot does not need to hit every intermediate
waypoint exactly.

### Strengths And Weaknesses

Flow 5 is the most reliable flow for static maps because A* handles global
route planning deterministically. The model no longer has to infer a safe
corridor through walls from the image. It mostly follows a line with computed
bearing guidance.

The tradeoff is that flow 5 does not use LiDAR and does not replan every cycle.
If the map is wrong, a dynamic obstacle appears, or the robot deviates far from
the path, the LLM can only choose movement tools using the current suggestion
and image. It cannot request a new global plan unless the code is extended to
support replanning.

## Comparison

| Flow | Core Design | Perception | LLM Freedom | Main Benefit | Main Risk |
|---|---|---|---|---|---|
| 1 | Hand-written loop | Map image injected as human message | High | Simple and transparent | Model controls tool order and can drift |
| 2 | LangGraph ReAct | Map image returned in ToolMessage | High | Cleaner ReAct implementation | Still relies on prompt for look-act discipline |
| 3 | LangGraph ReAct + LiDAR | Map plus raw/interpreted LiDAR | High | Better short-range safety awareness | More prompt/tool complexity and more calls |
| 4 | Fixed gather-decide-execute-check | Python always gathers map + LiDAR | Medium | One observation and one action per cycle | LLM still performs visual local planning |
| 5 | A* planner + LLM follower | Planned path image + computed bearing | Low | Most deterministic on static maps | No LiDAR or per-cycle replanning |

## Evolution Across Flows

The flows show a progression from maximum LLM autonomy toward more deterministic
robotics control:

1. Flow 1 gives the model direct control over all tools and manually injects
   map images.
2. Flow 2 standardizes the loop with LangGraph and sends images as true
   multimodal tool results.
3. Flow 3 adds LiDAR so the model can account for live physical obstacles.
4. Flow 4 removes perception-tool choice and forces a fixed one-action cycle.
5. Flow 5 removes global path planning from the LLM and has it follow an A*
   path with computed steering hints.

In short, flows 1-3 test how well an LLM can behave as a free-form navigation
agent, flow 4 constrains it into a control loop, and flow 5 uses classical
planning for the route while keeping the LLM only as a high-level movement
selector.
