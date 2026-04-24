# `nav2_llm_demo`

Vision-based LLM agent that drives a TurtleBot3 robot toward a goal by **looking at an annotated top-down map** and calling movement tools (rotate, move forward). Bypasses Nav2 entirely and writes directly to `/cmd_vel`.

Two interchangeable agent **flows** are bundled. They share the ROS node, the renderer, the prompt structure, and the per-LLM-call logging format, but differ in *how* the map image reaches the LLM and *which* agent loop runs the conversation. Switch with one CLI flag.

---

## TL;DR

```bash
# Inside the Docker container
colcon build --packages-select nav2_llm_demo
source install/setup.bash

# Default flow (1) — custom loop, image via follow-up HumanMessage, OpenAI
bash src/nav2_llm_demo/scripts/run_llm_nav.sh warehouse

# Flow 2 — LangGraph create_react_agent, image-in-ToolMessage, Anthropic Claude
bash src/nav2_llm_demo/scripts/run_llm_nav.sh warehouse --flow 2
```

Each run writes per-LLM-call artifacts under:

```
intro/llm_agent_runs/flow_<N>/<timestamp>/llm_controls_call_NNN/
    ├── request.json     # exact messages sent to the LLM
    ├── response.json    # what the LLM replied
    └── image_sent.png   # the map image that was in the request
```

---

## How it works (shared, both flows)

1. The Gazebo `.world` referenced by the map's sidecar is started, and the TurtleBot3 is spawned at the **source pose** declared in `custom_map_builder/maps/map_poses.yaml`.
2. The matching PGM/YAML map is loaded into `map_server` and shown in RViz.
3. A static `map -> odom` TF is published using the sidecar's `world_to_map_offset`, so the robot appears at the correct map-frame coordinate at startup.
4. `llm_agent_node` subscribes to `/odom`, publishes `/cmd_vel`, and runs the chosen agent flow in a background thread.
5. The agent receives an annotated top-down map (robot = red dot + arrow, destination = green plus, source = blue circle) and decides what to do via tool calls.
6. Each `move_forward` / `rotate` is executed by the ROS node and the robot **stops on its own** when the requested distance/rotation is reached, then control returns to the LLM.
7. Navigation continues until the robot is within `goal_tolerance_m` of the destination or `max_agent_steps` is exhausted.

---

## The two flows

|   | flow_1 (default) | flow_2 |
|---|------------------|--------|
| **Loop** | Custom hand-written loop in [`flow_1/agent.py`](nav2_llm_demo/llm/flow_1/agent.py) | LangGraph prebuilt `create_react_agent` in [`flow_2/agent.py`](nav2_llm_demo/llm/flow_2/agent.py) |
| **How the image reaches the LLM** | After every `get_map_view`, the loop appends a separate `HumanMessage` carrying the image | `get_map_view` returns a multimodal `ToolMessage` containing the image directly — no follow-up message |
| **Image refresh** | Only when LLM explicitly calls `get_map_view` | Same — the system prompt strongly nudges the model to call `get_map_view` after every move/rotate |
| **Provider compat** | Any vision LLM (OpenAI, Anthropic, etc.) | Requires a provider that accepts images inside `tool_result` blocks — **Anthropic Claude** works, **OpenAI does NOT** (returns HTTP 400) |
| **Default model** | `LLM_PROVIDER` / `LLM_MODEL` from `.env` | `FLOW2_LLM_PROVIDER` / `FLOW2_LLM_MODEL` from `.env`, currently `anthropic` / `claude-sonnet-4-6` |
| **Log dir** | `llm_agent_runs/flow_1/<stamp>/` | `llm_agent_runs/flow_2/<stamp>/` |

### Why flow_2 needs Claude

OpenAI's API returns HTTP 400 for any `tool` role message that contains an `image_url`:

```
Invalid 'messages[3]'. Image URLs are only allowed for messages with role 'user',
but this message with role 'tool' contains an image URL.
```

Anthropic Claude accepts image content blocks inside `tool_result` natively, which is exactly what flow_2 produces. So the per-flow override pins flow_2 to Claude regardless of the global `.env` provider.

---

## LLM provider / model selection

`.env` (kept at `intro/.env`, gitignored) holds two layers:

```
# Global default for any flow that doesn't override.
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o

# Per-flow overrides. Flow N reads FLOW{N}_LLM_PROVIDER / FLOW{N}_LLM_MODEL first.
FLOW2_LLM_PROVIDER=anthropic
FLOW2_LLM_MODEL=claude-sonnet-4-6

OPENAI_API_KEY=sk-proj-...
ANTHROPIC_API_KEY=sk-ant-...
```

Resolution order (handled by `resolve_llm_config(flow)` in [`llm/controller.py`](nav2_llm_demo/llm/controller.py)):

1. `FLOW{N}_LLM_PROVIDER` / `FLOW{N}_LLM_MODEL`
2. `LLM_PROVIDER` / `LLM_MODEL`

The runner banner prints both so you can tell at a glance:

```
LLM (global default)  : openai/gpt-4o
LLM (effective flow)  : anthropic/claude-sonnet-4-6
```

### Currently valid Claude Sonnet model names (April 2026)

- `claude-sonnet-4-6` — latest, recommended
- `claude-sonnet-4-5-20250929` — stable, dated pin

Any model retired by Anthropic returns a 404 (`claude-3-5-sonnet-20241022` was retired 2025-10-28; `claude-3-7-sonnet-20250219` retired 2026-02-19). Check `https://docs.anthropic.com/en/docs/resources/model-deprecations` if you hit a 404.

### Required pip packages (inside Docker)

| Provider | Package |
|----------|---------|
| OpenAI | `pip install langchain-openai` |
| Anthropic | `pip install langchain-anthropic` |
| Mistral | `pip install langchain-mistralai` |
| Ollama | `pip install langchain-ollama` |

Plus the always-needed: `langchain`, `langchain-core`, `langgraph`, `Pillow`, `numpy`, `pyyaml`.

---

## Single source of truth: `map_poses.yaml`

This package reads the unified `custom_map_builder/maps/map_poses.yaml`. Each map entry looks like:

```yaml
maps:
  warehouse.pgm:
    sidecar: src/world_to_map/maps/warehouse.world_map.yaml
    source:
      position: {x: -4.7094, y: -3.8890, z: 0.0}    # GAZEBO WORLD frame
      orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}
      yaw_rad: 0.0
    destination:
      position: {x: 2.8222, y: 1.9102, z: 0.0}      # GAZEBO WORLD frame
      orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}
      yaw_rad: 0.0
```

Coordinates are always in the **Gazebo world frame** (because that's what `custom_map_builder`'s "GAZEBO WORLD" output gives you). The runner script ([`scripts/parse_map_poses.py`](scripts/parse_map_poses.py)) automatically:

- Reads the sidecar (`src/world_to_map/maps/<name>.world_map.yaml`) for the `world_to_map_offset` and the matching `.world` file path.
- Spawns the robot in Gazebo at `(source.x, source.y)` (world frame).
- Sets the `map -> odom` static TF to the sidecar offset so the robot's `map -> base_footprint` lands at the expected map-frame coordinate.
- Passes map-frame source/destination to `llm_agent_node` (which compares poses against `map -> base_link` TF lookups).

For **hand-crafted maps** (`sidecar: null`, e.g. `diamond_blocked`), the offset is `(0, 0)`, the YAML coords are interpreted as map-frame directly, and Gazebo is not started — the agent operates on the 2D map only.

---

## Running

```bash
bash src/nav2_llm_demo/scripts/run_llm_nav.sh <MAP_NAME> [--flow 1|2]
```

Examples:

```bash
bash src/nav2_llm_demo/scripts/run_llm_nav.sh warehouse                # flow 1, default
bash src/nav2_llm_demo/scripts/run_llm_nav.sh warehouse --flow 2       # flow 2, Anthropic Claude
LLM_FLOW=2 bash src/nav2_llm_demo/scripts/run_llm_nav.sh warehouse     # same, via env
bash src/nav2_llm_demo/scripts/run_llm_nav.sh diamond_blocked          # legacy hand-crafted map
```

The runner:

- Sources `/opt/ros/$ROS_DISTRO/setup.bash` and `install/setup.bash`.
- Sources your `.env` so `LLM_PROVIDER`, `LLM_MODEL`, `FLOW{N}_LLM_*`, and API keys propagate to every spawned process.
- Calls `parse_map_poses.py <name>` to resolve everything from the unified file: map yaml, world file, sidecar offset, spawn pose, source/dest in both frames.
- Auto-detects `GAZEBO_MODEL_PATH` from the asset collection in the workspace.
- Pre-flight checks the agent executable + Python imports so silent failures surface early.
- Launches `nav2_llm_demo llm_agent.launch.py` with all arguments wired up.

### CLI argument

| Flag | Purpose |
|------|---------|
| `--flow 1` (default) | Use the custom loop with HumanMessage(image) injection. |
| `--flow 2` | Use LangGraph `create_react_agent` with image-in-ToolMessage. |

### Environment overrides

| Variable | Purpose |
|----------|---------|
| `MAP_POSES_PATH` | Override path to `map_poses.yaml`. |
| `GAZEBO_MODEL_PATH` | Manually set instead of auto-detecting. |
| `LAUNCH_RVIZ=false` | Skip RViz. |
| `USE_SIM_TIME=false` | Use wall clock instead of `/clock`. |
| `TURTLEBOT3_MODEL` | `burger` (default) / `waffle` / `waffle_pi`. |
| `LLM_FLOW=1\|2` | Same as `--flow N`; CLI flag takes precedence. |
| `LLM_PROVIDER` / `LLM_MODEL` | Global default LLM. |
| `FLOW1_LLM_PROVIDER` / `FLOW1_LLM_MODEL` | Override for flow_1 only. |
| `FLOW2_LLM_PROVIDER` / `FLOW2_LLM_MODEL` | Override for flow_2 only. |

---

## Adding a new world

1. Generate the rasterized map: `bash src/world_to_map/run_world_to_map.sh <world_stem>` (creates `src/world_to_map/maps/<name>.{pgm,yaml,world_map.yaml}`).
2. Pick source/destination by clicking in RViz: `bash src/custom_map_builder/scripts/run_map_builder.sh <name>`. The terminal prints both `MAP` and `GAZEBO WORLD` coordinates per click.
3. Add an entry under `maps:` in `custom_map_builder/maps/map_poses.yaml` using the **GAZEBO WORLD** values. Reference the sidecar.
4. Build: `colcon build --packages-select nav2_llm_demo` (also rebuild `custom_map_builder` and `world_to_map` if you changed them).
5. Run: `bash src/nav2_llm_demo/scripts/run_llm_nav.sh <name>`.

If you want a quick visual sanity-check of how the map will look to the LLM **without spawning a robot**, use the `map_view_debug` package — click in RViz, the rendered `image_sent.png` lands on disk for inspection.

---

## Agent tools (exposed to the LLM)

| Tool | Description |
|------|-------------|
| `get_map_view()` | Captures an annotated top-down map image. flow_1 returns a text result + appends a follow-up `HumanMessage` with the image; flow_2 returns the image directly inside the `ToolMessage`. |
| `move_forward(distance_meters)` | Drives forward (positive) or backward (negative) via `/cmd_vel`. Robot stops on its own when the distance is reached. |
| `rotate(angle_degrees)` | Rotates in place; positive = CCW (left), negative = CW (right). Robot stops on its own. |
| `get_robot_pose()` | Returns current `(x, y, yaw_degrees)` in the map frame as JSON. |
| `check_goal_reached()` | Returns `GOAL REACHED ...` if within 0.5 m of destination, else `NOT_REACHED ...`. |

---

## Logging layout

Per-LLM-call artifacts are the only thing written under each run dir. Both flows use the same format so runs can be diffed side by side.

```
intro/llm_agent_runs/
├── flow_1/
│   └── 20260424_185500/
│       ├── llm_controls_call_001/
│       │   ├── request.json    # full messages array sent to the LLM
│       │   ├── response.json   # the LLM's reply (tool_calls or content)
│       │   └── image_sent.png  # the actual PNG that was in the request
│       ├── llm_controls_call_002/...
│       └── ...
└── flow_2/
    └── 20260424_194004/
        └── llm_controls_call_001/...
```

`request.json` always includes:

- `llm_call_num`, `agent_step`
- `model`, `provider` (the resolved per-flow values, not the env globals)
- `image_sent: "image_sent.png"` (or `null` when there was no image in the request)
- `messages` — a JSON-serialized version of every message in the LLM context. Inline base64 PNGs are replaced with `"<inline PNG, see image_sent.png>"` to keep the JSON readable.

---

## Architecture

```
run_llm_nav.sh
└── parse_map_poses.py  (map_poses.yaml + sidecar -> shell vars)
    └── ros2 launch nav2_llm_demo llm_agent.launch.py flow:=N ...
        ├── SetEnvironmentVariable(LLM_FLOW, N)
        ├── (sidecar flow) include world_to_map.launch.py
        │   ├── Gazebo (custom .world) + TurtleBot3 spawn at source pose
        │   ├── map_server (PGM/YAML)
        │   ├── static TF: map -> odom = sidecar.world_to_map_offset
        │   └── RViz
        └── llm_agent_node
            ├── subscribes: /odom
            ├── publishes:  /cmd_vel, /navigation_status
            ├── TF listener: map -> base_footprint / base_link
            └── agent (chosen by LLM_FLOW)
                ├── flow_1: VisionNavigationAgent (custom loop)
                │   └── HumanMessage(image) appended after get_map_view
                └── flow_2: Flow2Agent (LangGraph create_react_agent)
                    └── ToolMessage carries the image directly
```

Module map:

```
nav2_llm_demo/llm/
├── __init__.py            # dispatcher: picks build_agent by $LLM_FLOW
├── controller.py          # RobotController protocol + resolve_llm_config + make_run_dir
├── map_renderer.py        # PGM -> annotated PNG (shared)
├── flow_1/
│   ├── agent.py           # custom loop
│   └── prompt.py          # SYSTEM_PROMPT + MAP_IMAGE_CONTEXT
└── flow_2/
    ├── agent.py           # wraps create_react_agent
    ├── prompt.py          # ReAct prompt nudging get_map_view after every move
    ├── tools.py           # @tool defs; get_map_view returns multimodal content
    └── logging.py         # PerCallLogger callback (request.json/response.json/image_sent.png)
```

---

## Important notes

- The agent compares its pose to source/destination in the **map frame**. The runner script handles converting Gazebo-world-frame YAML coords to map-frame using the sidecar offset before passing them to the launch file.
- The robot's motion is real — physical collision with `.world` obstacles will block it, just like manual driving in Gazebo.
- Movement is monitored via odometry to ensure the requested distance/rotation is achieved (and times out via `move_timeout_sec`).
- `max_agent_steps` (default 50) caps the agent loop so a stuck LLM can't run forever. flow_2 also enforces a LangGraph `recursion_limit=100`.
- The legacy `nav_config.yaml` and `parse_nav_config.py` are left in the tree but unused by the runner; new maps belong in `custom_map_builder/maps/map_poses.yaml`.
