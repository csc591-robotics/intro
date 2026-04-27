"""Batch experiment orchestrator for nav2_llm_experiments.

Reads ``experiments.yaml``, takes ``--flow`` and ``--experiment`` filters
from the CLI, and runs each (experiment, flow) combo as a fully isolated
subprocess group:

    1. Materialize a temp ``map_poses.yaml`` overriding the experiment's
       map entry with the experiment's source/destination pose.
    2. Spawn the recorder node (subscribes to /odom, /scan, /cmd_vel,
       /navigation_status, TF; writes JSONL streams; touches done.flag
       when /navigation_status reports terminal completion).
    3. Spawn ``ros2 bag record`` (unless --no-rosbag).
    4. Spawn ``run_llm_nav.sh <map> --flow <F>`` in its own process group
       with these env vars:
         MAP_POSES_PATH=<temp yaml>
         LLM_RUN_DIR_OVERRIDE=<flow_dir>/llm_calls
         LLM_AGENT_MAX_STEPS=<experiment.max_steps or default>
         LLM_ACTIONS_LOG=<flow_dir>/actions.jsonl
       Stdout/stderr captured to agent_stdout.log / agent_stderr.log.
    5. Wait for done.flag (with a wall-clock safety cap), then SIGINT the
       launch process group so Gazebo + RViz + map_server + agent all
       tear down cleanly.
    6. Stop recorder + rosbag (SIGINT, then SIGTERM, then SIGKILL).
    7. Write metadata.json with outcome, distances, durations, env.

Layout:

    <workspace>/experiment_data_folder/
      experiment_<id>/
        flow_<F>/
          <YYYYMMDD_HHMMSS>/
            metadata.json
            experiment.yaml
            map_poses_used.yaml
            map/                  (snapshot of pgm + yaml + sidecar)
            llm_calls/            (existing per-flow per-call writer output)
            pose_stream.jsonl
            pose_map_stream.jsonl
            scan_stream.jsonl
            cmd_vel_stream.jsonl
            status.log
            actions.jsonl
            rosbag/
            agent_stdout.log
            agent_stderr.log
            done.flag             (recorder writes this on terminal status)
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import platform
import shutil
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


VALID_FLOWS = ("1", "2", "3", "4", "5", "6")

# When ``ros2`` is not on PATH, try these under ``/opt/ros/<name>/bin/`` first
# if multiple distributions are installed.
_ROS_DISTRO_TRY_ORDER = (
    "humble", "jazzy", "iron", "rolling", "galactic", "foxy",
)


# ============================================================================
# Helpers
# ============================================================================

def _resolve_workspace_root(start: Path) -> Path:
    """Walk up from ``start`` to find the colcon workspace (contains src/)."""
    cur = start.resolve()
    for _ in range(8):
        if (cur / "src").is_dir():
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    return start.resolve()


def _now_ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_id_list(arg: str | None) -> list[int] | None:
    """``"1,3,5"`` -> ``[1, 3, 5]``. Returns None when ``arg`` is None/empty."""
    if not arg:
        return None
    out: list[int] = []
    for chunk in arg.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            out.append(int(chunk))
        except ValueError:
            raise SystemExit(f"--experiment / --flow value not an integer: {chunk!r}")
    return out


def _parse_flow_list(arg: str | None) -> list[str]:
    """``"3,5"`` -> ``["3", "5"]``; default -> ``["5"]``. Validates flow ids."""
    if not arg:
        return ["5"]
    out: list[str] = []
    for chunk in arg.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if chunk not in VALID_FLOWS:
            raise SystemExit(
                f"--flow {chunk!r} not supported; valid: {', '.join(VALID_FLOWS)}"
            )
        if chunk not in out:
            out.append(chunk)
    if not out:
        return ["5"]
    return out


def _yaw_from_quat(qx: float, qy: float, qz: float, qw: float) -> float:
    siny = 2.0 * (qw * qz + qx * qy)
    cosy = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny, cosy)


def _experiment_yaw(pose_block: dict[str, Any]) -> float:
    """Match parse_map_poses.py's quat-or-yaw_rad fallback."""
    if not isinstance(pose_block, dict):
        return 0.0
    ori = pose_block.get("orientation")
    if isinstance(ori, dict):
        return _yaw_from_quat(
            float(ori.get("x", 0.0)), float(ori.get("y", 0.0)),
            float(ori.get("z", 0.0)), float(ori.get("w", 1.0)),
        )
    return float(pose_block.get("yaw_rad", 0.0))


# ============================================================================
# Map poses I/O (write a temp file the existing parser can consume)
# ============================================================================

def _load_yaml(p: Path) -> dict[str, Any]:
    with p.open() as fh:
        return yaml.safe_load(fh) or {}


def _candidate_map_keys(map_name: str) -> list[str]:
    if map_name.endswith(".pgm"):
        stem = map_name[:-4]
        return [map_name, stem]
    return [map_name + ".pgm", map_name]


def _materialize_temp_map_poses(
    map_poses_path: Path,
    experiment: dict[str, Any],
    out_path: Path,
) -> tuple[str, dict[str, Any]]:
    """Write a copy of ``map_poses.yaml`` with the experiment's map entry
    overridden to use the experiment's source/destination poses.

    Returns ``(map_key_used, full_dict_written)``.
    Raises SystemExit if the experiment's map is not present in
    map_poses.yaml.
    """
    data = _load_yaml(map_poses_path)
    maps = data.get("maps") or {}
    map_name = experiment["map"]
    chosen_key: str | None = None
    for cand in _candidate_map_keys(map_name):
        if cand in maps:
            chosen_key = cand
            break
    if chosen_key is None:
        raise SystemExit(
            f"experiment id={experiment.get('id')} references map "
            f"{map_name!r} which is not in {map_poses_path}. "
            f"Available: {sorted(maps.keys())}"
        )

    new_data = copy.deepcopy(data)
    entry = copy.deepcopy(maps[chosen_key])
    entry["source"] = copy.deepcopy(experiment["source"])
    entry["destination"] = copy.deepcopy(experiment["destination"])
    new_data["maps"][chosen_key] = entry

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        yaml.safe_dump(new_data, fh, sort_keys=False)
    return chosen_key, new_data


# ============================================================================
# Subprocess management
# ============================================================================

class ManagedProc:
    """Wrap subprocess.Popen with SIGINT-then-escalate teardown across the
    entire process group."""

    def __init__(
        self,
        name: str,
        argv: list[str],
        env: dict[str, str] | None = None,
        cwd: Path | None = None,
        stdout: Any = None,
        stderr: Any = None,
    ) -> None:
        self.name = name
        self.argv = argv
        self._stdout = stdout
        self._stderr = stderr
        self.proc = subprocess.Popen(
            argv,
            env=env,
            cwd=str(cwd) if cwd else None,
            stdout=stdout,
            stderr=stderr,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )

    @property
    def pid(self) -> int:
        return self.proc.pid

    @property
    def pgid(self) -> int:
        return os.getpgid(self.proc.pid)

    def poll(self) -> int | None:
        return self.proc.poll()

    def terminate_tree(
        self,
        sigint_grace_sec: float = 15.0,
        sigterm_grace_sec: float = 5.0,
    ) -> int | None:
        """SIGINT the whole pgrp, wait, escalate to SIGTERM, then SIGKILL."""
        if self.proc.poll() is not None:
            return self.proc.returncode
        try:
            pgid = os.getpgid(self.proc.pid)
        except ProcessLookupError:
            return self.proc.poll()

        for sig, grace in (
            (signal.SIGINT, sigint_grace_sec),
            (signal.SIGTERM, sigterm_grace_sec),
            (signal.SIGKILL, 2.0),
        ):
            try:
                os.killpg(pgid, sig)
            except ProcessLookupError:
                break
            t0 = time.monotonic()
            while time.monotonic() - t0 < grace:
                if self.proc.poll() is not None:
                    return self.proc.returncode
                time.sleep(0.2)
        return self.proc.poll()

    def close_files(self) -> None:
        for fh in (self._stdout, self._stderr):
            if fh is not None and hasattr(fh, "close"):
                try:
                    fh.close()
                except OSError:
                    pass


# ============================================================================
# Single experiment×flow run
# ============================================================================

def _snapshot_map_files(
    map_poses_path: Path,
    workspace: Path,
    map_key: str,
    full_yaml: dict[str, Any],
    out_dir: Path,
) -> None:
    """Copy the PGM + map YAML + sidecar into ``out_dir/map/`` so the run
    is reproducible even after map_poses.yaml evolves."""
    map_dir = out_dir / "map"
    map_dir.mkdir(parents=True, exist_ok=True)

    entry = (full_yaml.get("maps") or {}).get(map_key, {})
    sidecar_field = entry.get("sidecar") or ""
    sidecar_path: Path | None = None
    if sidecar_field:
        sp = Path(sidecar_field)
        if not sp.is_absolute():
            sp = workspace / sp
        if sp.is_file():
            sidecar_path = sp
            try:
                shutil.copy2(sp, map_dir / sp.name)
            except OSError:
                pass

    map_yaml_path: Path | None = None
    if sidecar_path is not None:
        try:
            sc = _load_yaml(sidecar_path)
            mp = sc.get("map_yaml") or ""
            if mp:
                p = Path(mp)
                if not p.is_absolute():
                    p = sidecar_path.parent / p
                if p.is_file():
                    map_yaml_path = p
        except OSError:
            pass

    if map_yaml_path is None:
        stem = map_key[:-4] if map_key.endswith(".pgm") else map_key
        for cand in (
            workspace / "src" / "nav2_llm_demo" / "maps" / f"{stem}.yaml",
            workspace / "src" / "custom_map_builder" / "maps" / f"{stem}.yaml",
        ):
            if cand.is_file():
                map_yaml_path = cand
                break

    if map_yaml_path is not None and map_yaml_path.is_file():
        try:
            shutil.copy2(map_yaml_path, map_dir / map_yaml_path.name)
        except OSError:
            pass
        try:
            sc = _load_yaml(map_yaml_path)
            pgm_field = sc.get("image") or ""
            if pgm_field:
                p = Path(pgm_field)
                if not p.is_absolute():
                    p = map_yaml_path.parent / p
                if p.is_file():
                    shutil.copy2(p, map_dir / p.name)
        except OSError:
            pass


def _write_experiment_yaml(experiment: dict[str, Any], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        yaml.safe_dump(experiment, fh, sort_keys=False)


def _wait_for_done_flag(
    done_flag: Path,
    launch_proc: ManagedProc,
    max_wallclock_sec: float,
) -> tuple[str, str]:
    """Block until done.flag exists, the launch process exits, or timeout.

    Returns ``(why, detail)``:
      - ``("done_flag", <reason>)`` if recorder produced one
      - ``("launch_exited", "rc=N")`` if the agent/launch died first
      - ``("timeout", "Xs")`` if the wall-clock cap hit
    """
    t0 = time.monotonic()
    while True:
        if done_flag.is_file():
            try:
                payload = json.loads(done_flag.read_text())
                reason = payload.get("reason", "")
            except (OSError, json.JSONDecodeError):
                reason = "(unparseable done.flag)"
            return ("done_flag", reason)
        rc = launch_proc.poll()
        if rc is not None:
            return ("launch_exited", f"rc={rc}")
        if time.monotonic() - t0 > max_wallclock_sec:
            return ("timeout", f"{max_wallclock_sec:.0f}s")
        time.sleep(0.5)


def _classify_outcome(
    why: str, detail: str, experiment_max_steps: int,
) -> str:
    if why == "timeout":
        return "timeout"
    if why == "launch_exited":
        return "crashed"
    if why == "done_flag":
        if "GOAL REACHED" in detail or "goal reached" in detail.lower():
            return "goal_reached"
        if "Agent exhausted" in detail or "Agent loop ended" in detail:
            return "max_steps_exhausted"
        if "Failed to build" in detail:
            return "agent_error"
        if "A* planner failed" in detail or "A* found no path" in detail:
            return "planner_failed"
        return "agent_error"
    return "unknown"


def _final_distance_to_goal(
    pose_map_jsonl: Path,
    pose_jsonl: Path,
    dest_xy: tuple[float, float],
) -> float | None:
    """Read the last pose entry and compute distance to dest. Prefers
    pose_map_stream (map frame) over /odom (odom frame)."""
    for path in (pose_map_jsonl, pose_jsonl):
        if not path.is_file():
            continue
        try:
            with path.open() as fh:
                last_line = ""
                for line in fh:
                    if line.strip():
                        last_line = line
            if not last_line:
                continue
            rec = json.loads(last_line)
            x = float(rec.get("x", 0.0))
            y = float(rec.get("y", 0.0))
            return math.hypot(dest_xy[0] - x, dest_xy[1] - y)
        except (OSError, json.JSONDecodeError, KeyError, TypeError):
            continue
    return None


def _count_jsonl_records(path: Path) -> int:
    if not path.is_file():
        return 0
    try:
        with path.open() as fh:
            return sum(1 for line in fh if line.strip())
    except OSError:
        return 0


def _count_llm_calls(llm_calls_dir: Path) -> int:
    if not llm_calls_dir.is_dir():
        return 0
    return sum(
        1 for p in llm_calls_dir.iterdir()
        if p.is_dir() and p.name.startswith("llm_controls_call_")
    )


def _git_commit(workspace: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(workspace),
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return ""


def _resolve_ros2_and_env(base: dict[str, str]) -> tuple[str, dict[str, str]]:
    """Return an absolute ``ros2`` path and *base* with PATH fixed if needed.

    ``Popen([\"ros2\", ...])`` raises :exc:`FileNotFoundError` when ROS was never
    sourced — e.g. ``python -m nav2_llm_experiments.run_experiments`` instead of
    ``run_experiments.sh``. We fall back to ``/opt/ros/<distro>/bin/ros2`` and
    prepend that ``bin`` to ``PATH`` so the recorder and rosbag children start.
    """
    path_for_which = base.get("PATH")
    ros2 = shutil.which("ros2", path=path_for_which)
    if ros2:
        return ros2, base

    distro = (base.get("ROS_DISTRO") or os.environ.get("ROS_DISTRO") or "").strip()

    candidates: list[Path] = []
    if distro:
        p = Path(f"/opt/ros/{distro}/bin/ros2")
        if p.is_file():
            candidates.append(p)

    if not candidates:
        try:
            root = Path("/opt/ros")
            if root.is_dir():
                seen_names: set[str] = set()
                for name in _ROS_DISTRO_TRY_ORDER:
                    p = root / name / "bin" / "ros2"
                    if p.is_file():
                        candidates.append(p)
                        seen_names.add(name)
                for child in sorted(root.iterdir(), key=lambda x: x.name):
                    if not child.is_dir() or child.name in seen_names:
                        continue
                    p = child / "bin" / "ros2"
                    if p.is_file():
                        candidates.append(p)
        except OSError:
            pass

    for candidate in candidates:
        bd = candidate.parent
        out = dict(base)
        sep = os.pathsep
        old_path = out.get("PATH", "")
        out["PATH"] = f"{bd}{sep}{old_path}" if old_path else str(bd)
        chosen = bd.parent.name
        if distro:
            out.setdefault("ROS_DISTRO", distro)
        elif chosen:
            out.setdefault("ROS_DISTRO", chosen)
        return str(candidate), out

    raise RuntimeError(
        "Could not find `ros2` on PATH or under /opt/ros/*/bin. "
        "Source ROS 2 (e.g. source /opt/ros/humble/setup.bash), "
        "source install/setup.bash, "
        "or run: bash src/nav2_llm_experiments/scripts/run_experiments.sh"
    )


# Process names / argv fragments to reap before spawning a fresh sim.
# Stale survivors from a Ctrl+C'd batch (or a crashed run) hold port 11345
# (gzserver), keep an old turtlebot3_burger entity alive (so spawn_entity
# fails with "already exists"), or hijack /odom and /cmd_vel from the
# previous experiment — making the new run's robot appear frozen at the
# *previous* run's last pose. We reap before every run_one() so each
# experiment starts from a clean slate.
_STALE_SIM_PATTERNS: tuple[str, ...] = (
    "gzserver",
    "gzclient",
    "rviz2",
    "llm_agent_node",
    "recorder_node",
    "ros2 bag record",
    "ros2 bag",
    "spawn_entity.py",
    "robot_state_publisher",
    "static_transform_publisher",
    "lifecycle_manager",
    "map_server",
    "run_llm_nav.sh",
)


def _reap_stale_simulators(verbose: bool = True) -> None:
    """Forcefully terminate leftover Gazebo/ROS sim processes from prior runs.

    We try ``pkill -f`` (matches against the full command line) then
    ``pkill`` (basename match) for each known pattern. Failures are
    silent — the most common case is "no such process", which is exactly
    what we want to ignore.
    """
    pkill = shutil.which("pkill")
    if not pkill:
        if verbose:
            print(
                "[orchestrator] pkill not on PATH; skipping pre-flight reap.",
                flush=True,
            )
        return

    reaped_any = False
    for pat in _STALE_SIM_PATTERNS:
        for argv in (
            [pkill, "-9", "-f", pat],
            [pkill, "-9", pat.split()[0]],
        ):
            try:
                rc = subprocess.run(
                    argv,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=5,
                ).returncode
            except (OSError, subprocess.SubprocessError):
                continue
            # pkill returns 0 if it killed something, 1 if nothing matched.
            if rc == 0:
                reaped_any = True
                if verbose:
                    print(
                        f"[orchestrator] reaped stale process matching: {pat}",
                        flush=True,
                    )
                break

    if reaped_any:
        # Give the OS a moment to release port 11345 + tear down shared mem.
        time.sleep(1.5)


def _resolve_flow_llm(flow: str) -> tuple[str, str]:
    if str(flow) == "6":
        # Flow 6 is the pure Nav2 baseline — no LLM is involved. Surface
        # that explicitly in metadata.json so downstream comparisons don't
        # accidentally treat it like an LLM run.
        return "nav2", "navigate_to_pose"
    provider = (
        os.environ.get(f"FLOW{flow}_LLM_PROVIDER", "").strip()
        or os.environ.get("LLM_PROVIDER", "").strip()
    )
    model = (
        os.environ.get(f"FLOW{flow}_LLM_MODEL", "").strip()
        or os.environ.get("LLM_MODEL", "").strip()
    )
    return provider, model


def _augment_env_for_local_pkg(
    env: dict[str, str], workspace: Path,
) -> dict[str, str]:
    """Make sure ``nav2_llm_experiments`` is discoverable by ``ros2`` children.

    When the orchestrator is launched from a shell that didn't fully source
    ``install/setup.bash`` (e.g. ``sudo su`` discards env, or the user
    forgot to source after ``colcon build``), ``AMENT_PREFIX_PATH`` may
    only contain ``/opt/ros/<distro>``. Then ``ros2 run nav2_llm_experiments
    recorder_node`` fails with::

        Package 'nav2_llm_experiments' not found

    and the orchestrator hangs forever waiting for ``done.flag`` (only the
    recorder writes it). We patch ``AMENT_PREFIX_PATH`` and ``PYTHONPATH``
    here so children always see the local install — regardless of how the
    operator's shell was set up.
    """
    install_dir = workspace / "install" / "nav2_llm_experiments"
    if not install_dir.is_dir():
        return env

    out = dict(env)
    sep = os.pathsep

    pkg_marker = (
        install_dir / "share" / "ament_index" / "resource_index"
        / "packages" / "nav2_llm_experiments"
    )
    if pkg_marker.is_file():
        prev = out.get("AMENT_PREFIX_PATH", "")
        if str(install_dir) not in prev.split(sep):
            out["AMENT_PREFIX_PATH"] = (
                f"{install_dir}{sep}{prev}" if prev else str(install_dir)
            )

    py_site = install_dir / "lib" / "python3.10" / "site-packages"
    if py_site.is_dir():
        prev = out.get("PYTHONPATH", "")
        if str(py_site) not in prev.split(sep):
            out["PYTHONPATH"] = (
                f"{py_site}{sep}{prev}" if prev else str(py_site)
            )

    bin_dir = install_dir / "lib" / "nav2_llm_experiments"
    if bin_dir.is_dir():
        prev = out.get("PATH", "")
        if str(bin_dir) not in prev.split(sep):
            out["PATH"] = f"{bin_dir}{sep}{prev}" if prev else str(bin_dir)

    return out


def _recorder_argv(
    workspace: Path, ros2_exe: str, output_dir: Path,
) -> list[str]:
    """Return argv to spawn the recorder node.

    Prefers invoking the installed entry-point script directly (which only
    needs ``PYTHONPATH`` to be correct), and falls back to ``ros2 run``
    when the install layout is unfamiliar.
    """
    direct = (
        workspace / "install" / "nav2_llm_experiments"
        / "lib" / "nav2_llm_experiments" / "recorder_node"
    )
    if direct.is_file() and os.access(direct, os.X_OK):
        return [
            sys.executable, str(direct),
            "--ros-args",
            "-p", f"output_dir:={output_dir}",
        ]

    return [
        ros2_exe, "run", "nav2_llm_experiments", "recorder_node",
        "--ros-args",
        "-p", f"output_dir:={output_dir}",
    ]


def run_one(
    experiment: dict[str, Any],
    flow: str,
    workspace: Path,
    map_poses_path: Path,
    output_root: Path,
    default_max_steps: int,
    no_rosbag: bool,
    no_rviz: bool,
    no_gazebo_gui: bool,
) -> dict[str, Any]:
    """Run a single (experiment, flow) combo end-to-end. Returns the
    metadata dict that was written to disk."""
    exp_id = experiment.get("id")
    exp_name = experiment.get("name", f"experiment_{exp_id}")
    map_name = experiment["map"]
    max_steps = int(experiment.get("max_steps", default_max_steps))

    ts = _now_ts()
    flow_dir = output_root / f"experiment_{exp_id}" / f"flow_{flow}" / ts
    flow_dir.mkdir(parents=True, exist_ok=True)
    llm_calls_dir = flow_dir / "llm_calls"
    llm_calls_dir.mkdir(parents=True, exist_ok=True)

    # ---- 0. Reap stale sim processes from prior runs -------------------
    # Without this, a leftover gzserver from a previously Ctrl+C'd batch
    # holds port 11345; the new gzserver crashes; spawn_entity then talks
    # to the *old* gzserver where turtlebot3_burger already exists; the new
    # agent's /cmd_vel goes nowhere; the user sees the robot stuck at the
    # *previous* experiment's last pose (often the reverse-route pose).
    _reap_stale_simulators(verbose=True)

    # ---- 1. Materialize temp map_poses.yaml ----------------------------
    temp_yaml = flow_dir / "map_poses_used.yaml"
    map_key, full_yaml = _materialize_temp_map_poses(
        map_poses_path, experiment, temp_yaml,
    )
    _write_experiment_yaml(experiment, flow_dir / "experiment.yaml")
    _snapshot_map_files(map_poses_path, workspace, map_key, full_yaml, flow_dir)

    ros2_exe, proc_env = _resolve_ros2_and_env(os.environ.copy())
    proc_env = _augment_env_for_local_pkg(proc_env, workspace)

    # ---- 2. Spawn recorder ---------------------------------------------
    recorder_log = (flow_dir / "recorder.log").open("w")
    recorder = ManagedProc(
        "recorder",
        argv=_recorder_argv(workspace, ros2_exe, flow_dir),
        env=proc_env,
        stdout=recorder_log,
        stderr=subprocess.STDOUT,
    )

    # ---- 3. Spawn rosbag recorder --------------------------------------
    bag_proc: ManagedProc | None = None
    if not no_rosbag:
        bag_dir = flow_dir / "rosbag"
        # ros2 bag will create the dir; we must NOT pre-create it.
        if bag_dir.exists():
            shutil.rmtree(bag_dir)
        bag_log = (flow_dir / "rosbag.log").open("w")
        bag_proc = ManagedProc(
            "rosbag",
            argv=[
                ros2_exe, "bag", "record",
                "-o", str(bag_dir),
                "/odom", "/scan", "/cmd_vel",
                "/tf", "/tf_static",
                "/navigation_status",
                "/clock", "/map",
            ],
            env=proc_env,
            stdout=bag_log,
            stderr=subprocess.STDOUT,
        )

    # Tiny sleep so recorder + bag are subscribed before the agent comes
    # up. Without this we miss the first /odom messages.
    time.sleep(2.0)

    # ---- 4. Spawn the launch tree (run_llm_nav.sh) ---------------------
    runner_path = (
        workspace / "src" / "nav2_llm_demo" / "scripts" / "run_llm_nav.sh"
    )
    if not runner_path.is_file():
        # Tear down what's running and bail.
        recorder.terminate_tree()
        recorder.close_files()
        if bag_proc:
            bag_proc.terminate_tree()
            bag_proc.close_files()
        raise SystemExit(f"run_llm_nav.sh not found at {runner_path}")

    env = proc_env.copy()
    env["MAP_POSES_PATH"] = str(temp_yaml)
    env["LLM_RUN_DIR_OVERRIDE"] = str(llm_calls_dir)
    env["LLM_AGENT_MAX_STEPS"] = str(max_steps)
    env["LLM_ACTIONS_LOG"] = str(flow_dir / "actions.jsonl")
    if no_rviz:
        env["LAUNCH_RVIZ"] = "false"
    if no_gazebo_gui:
        env["GAZEBO_GUI"] = "false"
        env["GAZEBO_HEADLESS"] = "true"

    agent_stdout = (flow_dir / "agent_stdout.log").open("w")
    agent_stderr = (flow_dir / "agent_stderr.log").open("w")
    started_iso = _now_iso()
    started_wall = time.time()
    launch = ManagedProc(
        "launch",
        argv=["bash", str(runner_path), map_name, "--flow", flow],
        env=env,
        cwd=workspace,
        stdout=agent_stdout,
        stderr=agent_stderr,
    )

    # ---- 5. Wait for completion ----------------------------------------
    safety_cap_sec = max(180.0, max_steps * 60.0)
    why, detail = _wait_for_done_flag(
        flow_dir / "done.flag", launch, safety_cap_sec,
    )
    ended_iso = _now_iso()
    ended_wall = time.time()

    # ---- 6. Tear down launch tree --------------------------------------
    launch_rc = launch.terminate_tree(
        sigint_grace_sec=15.0, sigterm_grace_sec=5.0,
    )
    launch.close_files()

    # ---- 7. Stop recorder + rosbag (rosbag MUST get SIGINT to flush) ---
    if bag_proc is not None:
        bag_proc.terminate_tree(
            sigint_grace_sec=10.0, sigterm_grace_sec=3.0,
        )
        bag_proc.close_files()
    recorder.terminate_tree(sigint_grace_sec=5.0, sigterm_grace_sec=2.0)
    recorder.close_files()

    # ---- 8. Compute outcome metadata -----------------------------------
    outcome = _classify_outcome(why, detail, max_steps)

    src_pos = (experiment.get("source") or {}).get("position", {}) or {}
    dst_pos = (experiment.get("destination") or {}).get("position", {}) or {}
    src_yaw = _experiment_yaw(experiment.get("source") or {})
    dst_yaw = _experiment_yaw(experiment.get("destination") or {})
    sidecar_field = (
        ((full_yaml.get("maps") or {}).get(map_key) or {}).get("sidecar") or ""
    )

    map_offset_x = 0.0
    map_offset_y = 0.0
    if sidecar_field:
        sp = Path(sidecar_field)
        if not sp.is_absolute():
            sp = workspace / sp
        if sp.is_file():
            try:
                sc = _load_yaml(sp)
                offset = sc.get("world_to_map_offset") or [0.0, 0.0]
                map_offset_x = float(offset[0])
                map_offset_y = float(offset[1])
            except (OSError, yaml.YAMLError, ValueError, TypeError):
                pass

    dest_map_xy = (
        float(dst_pos.get("x", 0.0)) + map_offset_x,
        float(dst_pos.get("y", 0.0)) + map_offset_y,
    )
    final_dist = _final_distance_to_goal(
        flow_dir / "pose_map_stream.jsonl",
        flow_dir / "pose_stream.jsonl",
        dest_map_xy,
    )

    provider, model = _resolve_flow_llm(flow)
    metadata = {
        "experiment_id": exp_id,
        "experiment_name": exp_name,
        "map": map_name,
        "map_key": map_key,
        "flow": int(flow),
        "llm_provider": provider,
        "llm_model": model,
        "source_world": {
            "x": float(src_pos.get("x", 0.0)),
            "y": float(src_pos.get("y", 0.0)),
            "z": float(src_pos.get("z", 0.0)),
            "yaw_rad": src_yaw,
        },
        "destination_world": {
            "x": float(dst_pos.get("x", 0.0)),
            "y": float(dst_pos.get("y", 0.0)),
            "z": float(dst_pos.get("z", 0.0)),
            "yaw_rad": dst_yaw,
        },
        "destination_map": {
            "x": dest_map_xy[0],
            "y": dest_map_xy[1],
        },
        "world_to_map_offset": [map_offset_x, map_offset_y],
        "max_steps": max_steps,
        "started_at": started_iso,
        "ended_at": ended_iso,
        "wall_clock_sec": round(ended_wall - started_wall, 3),
        "completion_signal": why,
        "completion_detail": detail,
        "outcome": outcome,
        "final_distance_to_goal_m": (
            round(final_dist, 3) if final_dist is not None else None
        ),
        "total_llm_cycles": _count_llm_calls(llm_calls_dir),
        "total_actions": _count_jsonl_records(flow_dir / "actions.jsonl"),
        "total_pose_samples_odom": _count_jsonl_records(
            flow_dir / "pose_stream.jsonl"
        ),
        "total_pose_samples_map": _count_jsonl_records(
            flow_dir / "pose_map_stream.jsonl"
        ),
        "total_scan_samples": _count_jsonl_records(
            flow_dir / "scan_stream.jsonl"
        ),
        "total_cmd_vel_samples": _count_jsonl_records(
            flow_dir / "cmd_vel_stream.jsonl"
        ),
        "rosbag_recorded": (not no_rosbag)
            and (flow_dir / "rosbag").is_dir(),
        "rviz_disabled": bool(no_rviz),
        "gazebo_gui_disabled": bool(no_gazebo_gui),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "ros_distro": os.environ.get("ROS_DISTRO", ""),
        "git_commit": _git_commit(workspace),
        "launch_rc": launch_rc,
    }

    try:
        (flow_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    except OSError:
        pass

    return metadata


# ============================================================================
# CLI
# ============================================================================

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run a batch of LLM navigation experiments.",
    )
    p.add_argument(
        "--config", default="",
        help="Path to experiments.yaml. Default: "
             "src/nav2_llm_experiments/config/experiments.yaml",
    )
    p.add_argument(
        "--map-poses", default="",
        help="Path to map_poses.yaml. Default: "
             "src/custom_map_builder/maps/map_poses.yaml",
    )
    p.add_argument(
        "--flow", default="",
        help="Comma-separated flow ids to run per experiment "
             "(e.g. '3,5'). Default: 5",
    )
    p.add_argument(
        "--experiment", default="",
        help="Comma-separated experiment ids to run "
             "(e.g. '1,5'). Default: all in experiments.yaml",
    )
    p.add_argument(
        "--output-dir", default="",
        help="Where experiment_data_folder/ lives. Default: "
             "<workspace>/experiment_data_folder",
    )
    p.add_argument(
        "--no-rosbag", action="store_true",
        help="Skip ros2 bag recording for each experiment.",
    )
    p.add_argument(
        "--no-rviz", action="store_true",
        help="Pass LAUNCH_RVIZ=false to run_llm_nav.sh (no RViz window).",
    )
    p.add_argument(
        "--no-gazebo-gui", action="store_true",
        help="Pass GAZEBO_GUI=false (gzserver only, no gzclient window).",
    )
    return p


def main() -> None:
    args = _build_arg_parser().parse_args()

    here = Path(__file__).resolve()
    workspace = _resolve_workspace_root(here)

    config_path = Path(args.config) if args.config else (
        workspace / "src" / "nav2_llm_experiments" / "config" / "experiments.yaml"
    )
    if not config_path.is_file():
        raise SystemExit(f"experiments.yaml not found: {config_path}")

    map_poses_path = Path(args.map_poses) if args.map_poses else (
        workspace / "src" / "custom_map_builder" / "maps" / "map_poses.yaml"
    )
    if not map_poses_path.is_file():
        raise SystemExit(f"map_poses.yaml not found: {map_poses_path}")

    output_root = Path(args.output_dir) if args.output_dir else (
        workspace / "experiment_data_folder"
    )
    output_root.mkdir(parents=True, exist_ok=True)

    cfg = _load_yaml(config_path)
    defaults = cfg.get("defaults") or {}
    default_max_steps = int(defaults.get("max_steps", 50))
    all_experiments = cfg.get("experiments") or []
    if not all_experiments:
        raise SystemExit(f"No experiments defined in {config_path}")

    selected_ids = _parse_id_list(args.experiment)
    selected_flows = _parse_flow_list(args.flow)

    if selected_ids is not None:
        id_set = set(selected_ids)
        selected = [e for e in all_experiments if e.get("id") in id_set]
        missing = id_set - {e.get("id") for e in selected}
        if missing:
            raise SystemExit(
                f"--experiment ids not in {config_path}: "
                f"{sorted(missing)}"
            )
    else:
        selected = list(all_experiments)

    print("=" * 72)
    print("nav2_llm_experiments — batch runner")
    print("=" * 72)
    print(f"  Config        : {config_path}")
    print(f"  map_poses     : {map_poses_path}")
    print(f"  Output root   : {output_root}")
    print(f"  Workspace     : {workspace}")
    print(
        "  Experiments   : "
        + ", ".join(str(e.get("id")) for e in selected)
    )
    print(f"  Flows         : {', '.join(selected_flows)}")
    print(f"  default_max   : {default_max_steps}")
    print(f"  rosbag        : {'off' if args.no_rosbag else 'on'}")
    print(f"  rviz          : {'off' if args.no_rviz else 'on'}")
    print(f"  gazebo gui    : {'off' if args.no_gazebo_gui else 'on'}")
    print("=" * 72)

    summary: list[dict[str, Any]] = []
    total = len(selected) * len(selected_flows)
    idx = 0
    for exp in selected:
        for flow in selected_flows:
            idx += 1
            exp_id = exp.get("id")
            print()
            print("-" * 72)
            print(
                f"[{idx}/{total}] experiment={exp_id} "
                f"({exp.get('name', '')}) flow={flow}"
            )
            print("-" * 72)
            try:
                meta = run_one(
                    experiment=exp,
                    flow=flow,
                    workspace=workspace,
                    map_poses_path=map_poses_path,
                    output_root=output_root,
                    default_max_steps=default_max_steps,
                    no_rosbag=args.no_rosbag,
                    no_rviz=args.no_rviz,
                    no_gazebo_gui=args.no_gazebo_gui,
                )
                outcome = meta.get("outcome", "?")
                dist = meta.get("final_distance_to_goal_m")
                print(
                    f"  -> outcome={outcome} "
                    f"final_dist={dist} "
                    f"cycles={meta.get('total_llm_cycles')}"
                )
                summary.append({
                    "experiment_id": exp_id,
                    "flow": int(flow),
                    "outcome": outcome,
                    "final_distance_to_goal_m": dist,
                    "total_llm_cycles": meta.get("total_llm_cycles"),
                    "wall_clock_sec": meta.get("wall_clock_sec"),
                })
            except Exception as exc:  # noqa: BLE001
                print(
                    f"  !! experiment {exp_id} flow {flow} crashed: "
                    f"{type(exc).__name__}: {exc}",
                    file=sys.stderr,
                )
                summary.append({
                    "experiment_id": exp_id,
                    "flow": int(flow),
                    "outcome": "orchestrator_error",
                    "error": f"{type(exc).__name__}: {exc}",
                })

    summary_path = output_root / f"_batch_summary_{_now_ts()}.json"
    try:
        summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    except OSError:
        pass

    print()
    print("=" * 72)
    print(f"Batch complete. Summary -> {summary_path}")
    print("=" * 72)
    for row in summary:
        print(json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()
