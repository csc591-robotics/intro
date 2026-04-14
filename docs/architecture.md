# Architecture

## Overview

`nav2_llm_demo` is a ROS 2 (Humble) package that adds a high-level LLM planning layer on top of Nav2. The LLM decides *which* checkpoints the robot should visit and in *which order*; Nav2 handles all low-level path planning and obstacle avoidance.

```
 ┌────────────────────────────────────────────────────────────┐
 │  Docker container (Ubuntu 22.04 + ROS 2 Humble + Gazebo)  │
 │                                                            │
 │  ┌──────────────┐   mission request   ┌────────────────┐  │
 │  │  ROS topic   │ ──────────────────► │  LlmNavNode    │  │
 │  │ /navigation  │                     │  (ROS 2 node)  │  │
 │  │ _request     │                     └───────┬────────┘  │
 │  └──────────────┘                             │            │
 │                                               │ plan_route │
 │  ┌──────────────┐   status updates    ┌───────▼────────┐  │
 │  │  ROS topic   │ ◄────────────────── │  LLM routing   │  │
 │  │ /navigation  │                     │  (LangChain)   │  │
 │  │ _status      │                     └───────┬────────┘  │
 │  └──────────────┘                             │            │
 │                                               │ HTTP/gRPC  │
 │  ┌──────────────┐   PoseStamped goals ┌───────▼────────┐  │
 │  │    Nav2      │ ◄────────────────── │  LLM provider  │  │
 │  │  (AMCL +     │                     │  (OpenAI /     │  │
 │  │  planners)   │                     │   Anthropic /  │  │
 │  └──────┬───────┘                     │   Mistral /    │  │
 │         │ /cmd_vel                    │   Ollama …)    │  │
 │  ┌──────▼───────┐                     └────────────────┘  │
 │  │  TurtleBot3  │                                          │
 │  │  (Gazebo)    │                                          │
 │  └──────────────┘                                          │
 └────────────────────────────────────────────────────────────┘
```

---

## Components

### `LlmNavNode` (`llm_nav_node.py`)

A ROS 2 `Node` that owns the mission lifecycle:

| Responsibility | Detail |
|---|---|
| Accept missions | Subscribes to `/navigation_request` (String) |
| Plan routes | Calls `plan_route()` from the LLM routing module |
| Execute routes | Sends one `PoseStamped` at a time to Nav2 via `BasicNavigator` |
| Monitor progress | Detects stalls and navigation timeouts per segment |
| Replan on failure | Marks failed edges as blocked and asks the LLM for a new route |
| Publish status | Sends human-readable updates to `/navigation_status` |

ROS parameters (set via `llm_nav_params.yaml` or `-p` flags):

| Parameter | Default | Description |
|---|---|---|
| `request_topic` | `/navigation_request` | Topic to receive mission strings |
| `status_topic` | `/navigation_status` | Topic for status updates |
| `active_goal_pose_topic` | `/active_goal_pose` | Publishes the current Nav2 goal pose |
| `map_frame` | `map` | TF frame for goal poses |
| `navigation_timeout_sec` | `90.0` | Hard timeout per segment |
| `stall_timeout_sec` | `15.0` | Cancel segment if no progress for this long |
| `stall_min_progress_m` | `0.15` | Minimum distance that counts as progress |
| `max_replans` | `3` | Abort mission after this many replanning cycles |
| `max_decision_attempts` | `2` | LLM retry attempts per planning call |
| `route_graph_path` | *(required)* | Absolute path to `route_graph.json` |
| `planner_notes` | *(see yaml)* | Free-text instructions appended to system prompt |

---

### LLM Routing Module (`llm/llm_routing.py`)

Pure-Python, framework-independent module with no ROS imports. Key functions:

| Function | Purpose |
|---|---|
| `load_route_graph()` | Loads and validates the checkpoint graph JSON |
| `plan_route()` | Retry loop — calls `make_decision()` and validates each attempt |
| `make_decision()` | Builds prompts, calls the LLM via LangChain, parses JSON response |
| `validate_decision()` | Ensures the returned route is legal within the graph |
| `build_decision_context()` | Assembles the JSON context sent to the LLM |

#### LangChain integration

The module uses [`init_chat_model`](https://python.langchain.com/docs/how_to/chat_models_universal_init/) to remain provider-agnostic:

```python
from langchain.chat_models import init_chat_model

llm = init_chat_model(
    model=os.environ["LLM_MODEL"],
    model_provider=os.environ["LLM_PROVIDER"],
    temperature=0.2,
)
response = llm.invoke([SystemMessage(...), HumanMessage(...)])
```

No Groq or any other provider SDK is imported directly. Swapping providers requires only changing `.env`.

---

### Route Graph (`config/route_graph.json`)

A static JSON map of the navigation environment:

```
start ──► south_staging ──► south_pass ──► goal_zone
      └──► north_staging ──► north_pass ──┘
```

Schema:

```json
{
  "start_checkpoint": "start",
  "checkpoints": {
    "<name>": { "x": float, "y": float, "yaw": float, "description": string }
  },
  "edges": [{ "from": string, "to": string }],
  "goal_aliases": { "<alias>": [string] }
}
```

The LLM only sees descriptions and the edge list — never raw coordinates. Coordinates are resolved by `LlmNavNode` after the route is validated.

---

## Environment Configuration

All LLM credentials live in `.env` (gitignored). Copy `.env.example` and fill in your values.

| Variable | Required | Description |
|---|---|---|
| `LLM_PROVIDER` | Yes | LangChain provider name (`openai`, `anthropic`, `mistralai`, `ollama`, …) |
| `LLM_MODEL` | Yes | Model identifier for that provider (e.g. `gpt-4o`, `claude-3-5-sonnet-20241022`) |
| `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` / … | Yes* | Provider-specific API key (*not needed for Ollama) |

### Supported providers (examples)

| Provider | `LLM_PROVIDER` | `LLM_MODEL` example | Extra package |
|---|---|---|---|
| OpenAI | `openai` | `gpt-4o` | `pip install langchain-openai` |
| Anthropic | `anthropic` | `claude-3-5-sonnet-20241022` | `pip install langchain-anthropic` |
| Mistral | `mistralai` | `mistral-large-latest` | `pip install langchain-mistralai` |
| Ollama (local) | `ollama` | `llama3.2` | `pip install langchain-ollama` |

Install the extra package inside the container before running `colcon build`.

---

## Data Flow

1. A string is published to `/navigation_request` (e.g. `"Reach the far side"`).
2. `LlmNavNode._handle_request()` stores the goal and clears mission state.
3. `_mission_tick()` fires every 1 s. On the first tick with no active route it calls `plan_route()`.
4. `make_decision()` builds a system prompt (with `planner_notes`) and a user JSON payload (graph context, blocked edges, current checkpoint) and sends them to the LLM via LangChain.
5. The LLM returns JSON: `{"goal_alias": "...", "route": [...], "reason": "..."}`.
6. `validate_decision()` checks every node and edge against the graph. If invalid, the loop retries up to `max_decision_attempts` times.
7. The validated route is stored; `_start_next_segment()` sends the first `PoseStamped` to Nav2.
8. `_tick_active_segment()` polls Nav2 feedback. On success it advances to the next segment. On failure/stall/timeout it marks the edge blocked and triggers a replan (back to step 4).
9. When the final checkpoint is reached, `_publish_status("Mission complete …")` fires and mission state is cleared.

---

## Launch

```bash
# Inside the Docker container
bash ./run_llm_nav.sh "Reach the far side of the obstacle course"
```

`run_llm_nav.sh` orchestrates:
1. Gazebo + TurtleBot3 simulation
2. Nav2 stack (AMCL, planners, costmaps)
3. `llm_nav_node` (this package)
4. Background `ros2 topic echo /navigation_status`
5. Initial mission publish to `/navigation_request`
