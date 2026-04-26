# `nav2_llm_experiments`

Batch experiment runner that sits on top of [`nav2_llm_demo`](../nav2_llm_demo). You declare experiments (map + start pose + end pose + optional `max_steps`) in a YAML file, pick which flow(s) and which experiment ids to run from the CLI, and the orchestrator spins up Gazebo + RViz + the LLM agent for each experiment, captures everything it can (per-LLM-call request/response/image, pose, lidar, cmd_vel, actions, rosbag2, outcome), tears it all down, then moves on to the next combo.

The standalone demo (`bash run_llm_nav.sh warehouse --flow 5`) is unchanged. Experiments reuse it as the underlying mechanism.

---

## TL;DR

```bash
# Inside the Docker container
colcon build --packages-select nav2_llm_demo nav2_llm_experiments
source install/setup.bash

# Run all experiments for flow 5 (default)
bash src/nav2_llm_experiments/scripts/run_experiments.sh

# Run all experiments for flows 3 AND 5 (cross product)
bash src/nav2_llm_experiments/scripts/run_experiments.sh --flow 3,5

# Run only experiments id=1 and id=5, just for flow 5, no RViz, no rosbag
bash src/nav2_llm_experiments/scripts/run_experiments.sh \
    --experiment 1,5 --flow 5 --no-rviz --no-rosbag
```

Each (experiment, flow) combo writes to:

```
intro/experiment_data_folder/experiment_<N>/flow_<F>/<YYYYMMDD_HHMMSS>/
    metadata.json
    experiment.yaml
    map_poses_used.yaml
    map/                  (pgm + yaml + sidecar snapshot)
    llm_calls/            (llm_controls_call_NNN/{request,response}.json + image_sent.png)
    pose_stream.jsonl
    pose_map_stream.jsonl
    scan_stream.jsonl
    cmd_vel_stream.jsonl
    actions.jsonl
    status.log
    rosbag/
    agent_stdout.log
    agent_stderr.log
    done.flag
```

---

## How it works

For every (experiment, flow) combo the orchestrator does this in order:

1. **Materialize a temp `map_poses.yaml`** — a copy of the workspace's [`custom_map_builder/maps/map_poses.yaml`](../custom_map_builder/maps/map_poses.yaml) with the experiment's `map` entry overridden so that `source` and `destination` come from the experiment file (the original `sidecar` field, route_graph, and other entries are kept). Saved to the run's `map_poses_used.yaml`.
2. **Spawn the recorder** — [`recorder_node.py`](nav2_llm_experiments/recorder_node.py) subscribes to `/odom`, `/scan`, `/cmd_vel`, `/navigation_status`, and the `map -> base_link` TF; writes append-only JSONL streams; touches `done.flag` the moment it sees a terminal status string from the agent.
3. **Spawn `ros2 bag record`** — bags `/odom /scan /cmd_vel /tf /tf_static /navigation_status /clock /map` into `rosbag/` (unless `--no-rosbag`).
4. **Spawn `run_llm_nav.sh <map> --flow <F>`** as a fresh process group, with these env vars set:
   - `MAP_POSES_PATH=<temp yaml>` — points the existing parser at the override file.
   - `LLM_RUN_DIR_OVERRIDE=<flow_dir>/llm_calls` — redirects the existing per-LLM-call writers into the experiment folder. **Image logging works automatically** because every flow already writes `image_sent.png` per call (flow_1 directly, flows 2-5 via the shared `PerCallLogger`); we just redirect their output dir.
   - `LLM_AGENT_MAX_STEPS=<experiment.max_steps or 50>` — caps the agent loop.
   - `LLM_ACTIONS_LOG=<flow_dir>/actions.jsonl` — turns on per-action JSONL records (`move_forward` / `rotate` start/end timestamps, requested vs achieved distance/angle, pose before/after, `timed_out`).
   - `LAUNCH_RVIZ=false` if `--no-rviz`; `GAZEBO_GUI=false` if `--no-gazebo-gui`.
   - Stdout / stderr captured to `agent_stdout.log` / `agent_stderr.log`.
5. **Wait for `done.flag`** to appear, with a hard wall-clock cap of `max_steps × 60s` so a wedged Gazebo can't block the batch forever.
6. **SIGINT the launch process group** — this is exactly what Ctrl-C does today. `ros2 launch` propagates the signal to every child: `gzserver`, `gzclient`, `rviz2`, `map_server`, `lifecycle_manager`, the static TF publisher, and the agent. 15s grace, then SIGTERM, then SIGKILL.
7. **SIGINT the recorder + rosbag** the same way. Rosbag flushes the `.db3` on SIGINT — losing it would corrupt the bag.
8. **Write `metadata.json`** with the outcome, distances, durations, total LLM cycles, etc.
9. **Move on to the next combo**, even if this one crashed (each combo is wrapped in a try/except).

### Gazebo + RViz lifecycle

Each experiment is a fresh `ros2 launch` invocation, so **Gazebo and RViz fully open at the start of every experiment and fully close at the end**. They are not kept alive across experiments. Reasons:

- Different experiments may use different maps / `.world` files.
- Re-spawning the TurtleBot3 in the same Gazebo instance is unreliable (lingering TF, leftover plugins, race conditions on `spawn_entity`).
- Full teardown is what Ctrl-C already does — we just trigger it programmatically via SIGINT to the process group.

The cost is ~15-30s of Gazebo + Nav2 startup per experiment. For unattended batch runs use `--no-rviz --no-gazebo-gui`.

---

## CLI

```
bash src/nav2_llm_experiments/scripts/run_experiments.sh \
    [--config <path>] \
    [--map-poses <path>] \
    [--flow 1,2,3,4,5] \
    [--experiment 1,3,5] \
    [--output-dir <path>] \
    [--no-rosbag] \
    [--no-rviz] \
    [--no-gazebo-gui]
```

| Flag | Default | Notes |
|------|---------|-------|
| `--config` | `src/nav2_llm_experiments/config/experiments.yaml` | The experiments file. |
| `--map-poses` | `src/custom_map_builder/maps/map_poses.yaml` | Source of sidecar / world / map YAML lookups. |
| `--flow` | `5` | Comma-separated. Each experiment runs once per listed flow (cross product). Valid: `1,2,3,4,5`. |
| `--experiment` | all | Comma-separated experiment `id`s. Default = every entry in the config file. |
| `--output-dir` | `<workspace>/experiment_data_folder` | Top-level output root. |
| `--no-rosbag` | off (= bag is recorded) | Skip `ros2 bag record`. |
| `--no-rviz` | off (= RViz opens) | Sets `LAUNCH_RVIZ=false`. |
| `--no-gazebo-gui` | off (= gzclient opens) | Sets `GAZEBO_GUI=false` (gzserver only, no client window). |

The CLI matches your two filtering use cases:

```bash
# Just experiment 1 and 5 across flows 3 and 5
bash src/nav2_llm_experiments/scripts/run_experiments.sh --experiment 1,5 --flow 3,5
# -> 4 runs in this order:
#    experiment_1 / flow_3
#    experiment_1 / flow_5
#    experiment_5 / flow_3
#    experiment_5 / flow_5
```

---

## Defining experiments

[`config/experiments.yaml`](config/experiments.yaml) — same pose shape as `map_poses.yaml`, but each entry is a fresh navigation task. **No** `flow` field (flow comes from the CLI), **no** `sidecar` field (sidecar is reused from `map_poses.yaml` via the `map` name).

```yaml
frame: gazebo_world

defaults:
  max_steps: 50           # used when an experiment omits max_steps

experiments:

  - id: 1
    name: warehouse_default_route
    map: warehouse        # must exist in custom_map_builder/maps/map_poses.yaml
    max_steps: 60         # optional override
    source:
      position: {x: -4.7094, y: -3.8890, z: 0.0}
      orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}
      yaw_rad: 0.0
    destination:
      position: {x: 2.8222, y: 1.9102, z: 0.0}
      orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}
      yaw_rad: 0.0

  - id: 2
    name: diamond_blocked_default
    map: diamond_blocked
    source:      {position: {x: 13.10, y: 11.34, z: 0.0}, orientation: {...}, yaw_rad: 2.88}
    destination: {position: {x: 10.24, y: 19.17, z: 0.0}, orientation: {...}, yaw_rad: 1.62}
```

### Coordinate frame rules

Identical to [`map_poses.yaml`](../custom_map_builder/maps/map_poses.yaml):

- For **sidecar maps** (`warehouse`, `workshop_example`, ...): `position.{x,y,z}` is in the **Gazebo world frame**. The sidecar's `world_to_map_offset` is applied automatically.
- For **hand-crafted maps** (`diamond_blocked`, ...): `position.{x,y,z}` is in the **map frame** (world == map for these; offset is `(0, 0)`).

To grab coordinates: `LAUNCH_GAZEBO=false bash src/custom_map_builder/scripts/run_map_builder.sh <map>`, then in RViz press `P` (Publish Point), click, and the terminal prints both frames. Write the `GAZEBO WORLD` value.

### `max_steps`

- If the experiment provides `max_steps: N`, the agent is capped at N LLM cycles.
- Otherwise `defaults.max_steps` from the same YAML is used (50 by default).
- The orchestrator's wall-clock safety cap is `max_steps × 60s` (just to break out of a wedged Gazebo); the experiment normally ends much sooner via goal-reach or step exhaustion.

---

## What gets captured per experiment

| File | Contents | Notes |
|------|----------|-------|
| `metadata.json` | Outcome, source/dest (world + map frame), durations, total LLM cycles, total actions, host, ROS_DISTRO, git commit, launch_rc | Written at the end. |
| `experiment.yaml` | The exact experiment entry that ran | For reproducibility. |
| `map_poses_used.yaml` | The materialized override map_poses.yaml | Snapshot of what `parse_map_poses.py` consumed. |
| `map/` | Snapshot of the PGM, map YAML, sidecar | So the run can be replayed even after `map_poses.yaml` evolves. |
| `llm_calls/llm_controls_call_NNN/request.json` | Exact messages sent to the LLM | **Identical format** to existing `intro/llm_agent_runs/flow_*/<ts>/llm_controls_call_*/`. |
| `llm_calls/llm_controls_call_NNN/response.json` | LLM reply | Same. |
| `llm_calls/llm_controls_call_NNN/image_sent.png` | The annotated map image the LLM saw on that call | Same. |
| `pose_stream.jsonl` | `/odom` timeline (x, y, yaw, lin_vel, ang_vel) | Append-only JSONL; safe to tail. |
| `pose_map_stream.jsonl` | `map -> base_link` TF lookup at 10 Hz | This is the actual map-frame pose the agent reasons about. |
| `scan_stream.jsonl` | LaserScan downsampled to 8 angular sectors (min/max/mean per sector + overall min) | Full raw scan is in the rosbag. |
| `cmd_vel_stream.jsonl` | Every `Twist` published to `/cmd_vel` | Raw control signal. |
| `actions.jsonl` | One record per `move_forward`/`rotate` call: requested vs achieved distance/angle, start/end timestamps, pose before/after, `timed_out` | Comes from `LLM_ACTIONS_LOG` hook in `llm_agent_node.py`. |
| `status.log` | `/navigation_status` text stream with timestamps | Human-readable progress trail. |
| `rosbag/` | `ros2 bag record` of `/odom /scan /cmd_vel /tf /tf_static /navigation_status /clock /map` | Replayable in RViz. Disable with `--no-rosbag`. |
| `agent_stdout.log` / `agent_stderr.log` | Captured stdout/stderr of `run_llm_nav.sh` | Use to debug crashes. |
| `done.flag` | One-line JSON `{"reason": "...", "t_wall": ..., "iso": "..."}` | Touched by the recorder when it sees a terminal status. The orchestrator polls this. |
| `recorder.log` / `rosbag.log` | Stdout/stderr of the side processes | Mostly noise; useful when something fails. |

### `metadata.json` schema

```json
{
  "experiment_id": 1,
  "experiment_name": "warehouse_default_route",
  "map": "warehouse",
  "map_key": "warehouse.pgm",
  "flow": 5,
  "llm_provider": "anthropic",
  "llm_model": "claude-sonnet-4-6",
  "source_world":      {"x": -4.7094, "y": -3.8890, "z": 0.0, "yaw_rad": 0.0},
  "destination_world": {"x":  2.8222, "y":  1.9102, "z": 0.0, "yaw_rad": 0.0},
  "destination_map":   {"x":  ...,    "y":  ...},
  "world_to_map_offset": [8.69, 11.76],
  "max_steps": 60,
  "started_at": "2026-04-26T12:00:00Z",
  "ended_at":   "2026-04-26T12:04:31Z",
  "wall_clock_sec": 271.4,
  "completion_signal": "done_flag" | "launch_exited" | "timeout",
  "completion_detail": "GOAL REACHED after 18 steps! ...",
  "outcome": "goal_reached" | "max_steps_exhausted" | "agent_error"
            | "planner_failed" | "timeout" | "crashed",
  "final_distance_to_goal_m": 0.21,
  "total_llm_cycles": 18,
  "total_actions": 36,
  "total_pose_samples_odom": 5430,
  "total_pose_samples_map": 2710,
  "total_scan_samples": 1356,
  "total_cmd_vel_samples": 412,
  "rosbag_recorded": true,
  "rviz_disabled": false,
  "gazebo_gui_disabled": false,
  "host": "...",
  "platform": "...",
  "ros_distro": "humble",
  "git_commit": "...",
  "launch_rc": 0
}
```

After the batch finishes, a top-level summary is also written:

```
<output_dir>/_batch_summary_<YYYYMMDD_HHMMSS>.json
```

— one line per (experiment, flow) row with `outcome`, `final_distance_to_goal_m`, `total_llm_cycles`, `wall_clock_sec`.

---

## Backward compatibility with the demo

`bash src/nav2_llm_demo/scripts/run_llm_nav.sh warehouse --flow 5` is **unchanged**. None of the new env vars (`LLM_RUN_DIR_OVERRIDE`, `LLM_AGENT_MAX_STEPS`, `LLM_ACTIONS_LOG`) are set in the demo flow, so:

- `make_run_dir` falls back to today's `intro/llm_agent_runs/flow_5/<timestamp>/` path.
- `max_agent_steps` keeps its launch-arg default of `0` (unlimited).
- `actions.jsonl` is not written.
- Gazebo + RViz keep running until you Ctrl-C, exactly like before.

The two small additive changes in `nav2_llm_demo` are env-var-gated:

| Where | What |
|-------|------|
| [`nav2_llm_demo/llm/controller.py`](../nav2_llm_demo/nav2_llm_demo/llm/controller.py) `make_run_dir` | Honors `LLM_RUN_DIR_OVERRIDE` when set. |
| [`nav2_llm_demo/llm_agent_node.py`](../nav2_llm_demo/nav2_llm_demo/llm_agent_node.py) | `max_agent_steps` falls back to `LLM_AGENT_MAX_STEPS` env var when its launch-arg value is `0`. `move_forward` / `rotate` append to `LLM_ACTIONS_LOG` when set. |

---

## Common operations

```bash
# How many LLM calls happened in the last warehouse/flow_5 run?
ls intro/experiment_data_folder/experiment_1/flow_5/*/llm_calls | wc -l

# Tail status of a running experiment
tail -f intro/experiment_data_folder/experiment_1/flow_5/*/status.log

# Replay the rosbag in RViz
ros2 bag play intro/experiment_data_folder/experiment_1/flow_5/<ts>/rosbag

# Quick CSV of pose at 1 Hz (using jq)
jq -r '"\(.t_wall),\(.x),\(.y),\(.yaw)"' \
    intro/experiment_data_folder/experiment_1/flow_5/<ts>/pose_map_stream.jsonl \
    | awk 'NR==1 || NR%10==0'

# Per-action summary
jq -r '"\(.name)\t\(.args)\t\(.duration_sec)\t\(.timed_out)"' \
    intro/experiment_data_folder/experiment_1/flow_5/<ts>/actions.jsonl

# Batch outcome summary across all runs
cat intro/experiment_data_folder/_batch_summary_*.json | jq '.[] | {experiment_id, flow, outcome, final_distance_to_goal_m}'
```

---

## Troubleshooting

**"Map ... not found in map_poses.yaml"** — your experiment references a `map:` value that has no entry in [`map_poses.yaml`](../custom_map_builder/maps/map_poses.yaml). Add the map there first (sidecar + at least an empty source/destination is fine — the experiment's pose values override these).

**Gazebo doesn't tear down between experiments** — the orchestrator SIGINTs the launch process group; `gzserver` and `gzclient` should die within ~15s. If they linger, the orchestrator escalates to SIGTERM then SIGKILL after the grace period. If you see leaked `gz*` processes, run `pkill -9 gz` between batches and please file a note here.

**`done.flag` never appears** — the recorder watches `/navigation_status` for these substrings: `GOAL REACHED`, `Agent reports goal reached`, `Agent terminated`, `Agent exhausted`, `Agent loop ended`, `Failed to build LLM agent`, `A* planner failed`, `A* found no path`. If the agent flow you're testing prints something different, add it to `TERMINAL_STATUS_SUBSTRINGS` in [`recorder_node.py`](nav2_llm_experiments/recorder_node.py). The orchestrator's wall-clock safety cap (`max_steps × 60s`) will eventually rescue the batch even if `done.flag` is never written.

**`ros2 bag record` complains about an existing folder** — the orchestrator deletes any pre-existing `rosbag/` folder before bag records into it. If you see this error you probably hit a permissions issue; check that the `experiment_data_folder/` tree is writable by the user running the orchestrator (it might have been created by `root` from a prior containerized run).

**Image logging is missing in `llm_calls/`** — that means the agent flow you're using doesn't route through `make_run_dir()` or doesn't write `image_sent.png`. All five bundled flows do (`flow_1` writes directly, `flows 2-5` via the shared [`PerCallLogger`](../nav2_llm_demo/nav2_llm_demo/llm/flow_2/logging.py)). If you've added a custom flow, make sure it calls `controller.make_run_dir(flow="N")` and uses `PerCallLogger` (or its own writer that follows the same convention).

**Pre-flight `nav2_llm_demo.llm` import error** — the orchestrator delegates to `run_llm_nav.sh`, which itself runs the import pre-flight. The most common fix is `pip install langchain-anthropic` (or whatever provider matches your `.env`), then `colcon build --packages-select nav2_llm_demo` to re-link the entry point.
