#!/usr/bin/env python3
"""Make simple matplotlib plots from experiment_data_folder outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
os.environ.setdefault("MPLCONFIGDIR", str(SCRIPT_DIR / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(SCRIPT_DIR / ".cache"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FLOW_COLORS = {
    "1": "#1f77b4",
    "2": "#ff7f0e",
    "3": "#2ca02c",
    "4": "#d62728",
    "5": "#9467bd",
    "6": "#8c564b",
    "7": "#e377c2",
}

FLOW_DISPLAY = {
    "1": "flow 1",
    "2": "flow 2",
    "3": "flow 3",
    "4": "flow 4",
    "5": "A* + LLM Follower",
    "6": "Nav2 NavigateToPose Baseline",
    "7": "Topology Graph + LLM Route Planner",
}


def _load_json(path: Path) -> dict:
    with path.open() as fh:
        return json.load(fh)


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.is_file():
        return rows
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _find_run_dirs(root: Path) -> list[Path]:
    run_dirs: list[Path] = []
    for metadata in root.glob("experiment_*/flow_*/*/metadata.json"):
        run_dirs.append(metadata.parent)
    return sorted(run_dirs)


def _run_label(meta: dict) -> str:
    flow = str(meta["flow"])
    flow_label = FLOW_DISPLAY.get(flow, f"flow {flow}")
    return f"exp{meta['experiment_id']}-{flow_label}"


def _method_label(meta: dict) -> str:
    flow = str(meta["flow"])
    return FLOW_DISPLAY.get(flow, f"flow {flow}")


def _load_simple_yaml(path: Path) -> dict:
    data: dict = {}
    current_key = None
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("-") and current_key is not None:
            data.setdefault(current_key, []).append(float(line[1:].strip()))
            continue
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        current_key = key
        if not value:
            data[key] = []
            continue
        if value.startswith(("'", '"')) and value.endswith(("'", '"')):
            value = value[1:-1]
        if value in {"true", "false"}:
            data[key] = value == "true"
        else:
            try:
                data[key] = int(value)
            except ValueError:
                try:
                    data[key] = float(value)
                except ValueError:
                    data[key] = value
    return data


def _map_yaml_for_key(map_key: str) -> Path | None:
    candidates = [
        REPO_ROOT / "src/world_to_map/maps" / f"{map_key}.yaml",
        REPO_ROOT / "src/nav2_llm_demo/maps" / f"{map_key}.yaml",
        REPO_ROOT / "src/custom_map_builder/maps" / f"{map_key}.yaml",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _draw_map_background(ax, meta: dict) -> bool:
    map_key = str(meta.get("map", "")).strip()
    if not map_key:
        return False

    yaml_path = _map_yaml_for_key(map_key)
    if yaml_path is None:
        return False

    map_cfg = _load_simple_yaml(yaml_path)
    image_name = map_cfg.get("image")
    resolution = map_cfg.get("resolution")
    origin = map_cfg.get("origin")
    if not image_name or resolution is None or not isinstance(origin, list) or len(origin) < 2:
        return False

    image_path = (yaml_path.parent / str(image_name)).resolve()
    if not image_path.is_file():
        return False

    image = plt.imread(image_path)
    height, width = image.shape[:2]
    x0 = float(origin[0])
    y0 = float(origin[1])
    res = float(resolution)
    extent = [x0, x0 + width * res, y0, y0 + height * res]
    ax.imshow(image, cmap="gray", origin="lower", extent=extent, alpha=0.85)
    return True


def _path_length_m(run_dir: Path) -> float:
    pose_rows = _load_jsonl(run_dir / "pose_map_stream.jsonl")
    if len(pose_rows) < 2:
        return 0.0

    total = 0.0
    prev_x = None
    prev_y = None
    for row in pose_rows:
        if "x" not in row or "y" not in row:
            continue
        x = float(row["x"])
        y = float(row["y"])
        if prev_x is not None and prev_y is not None:
            total += math.hypot(x - prev_x, y - prev_y)
        prev_x = x
        prev_y = y
    return total


def _straight_line_goal_distance_m(meta: dict) -> float:
    source = meta.get("source_map", {})
    dest = meta.get("destination_map", {})
    if not isinstance(source, dict) or not isinstance(dest, dict):
        return 0.0
    if "x" not in source or "y" not in source or "x" not in dest or "y" not in dest:
        return 0.0
    return math.hypot(float(dest["x"]) - float(source["x"]), float(dest["y"]) - float(source["y"]))


def _plot_bar(
    labels: list[str],
    values: list[float],
    title: str,
    ylabel: str,
    output: Path,
) -> None:
    if not labels:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(labels, values)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def _plot_trajectory(run_dir: Path, meta: dict, output_dir: Path) -> None:
    pose_rows = _load_jsonl(run_dir / "pose_map_stream.jsonl")
    if not pose_rows:
        return

    xs = [row["x"] for row in pose_rows if "x" in row]
    ys = [row["y"] for row in pose_rows if "y" in row]
    if not xs or not ys:
        return

    dest = meta.get("destination_map", {})
    fig, ax = plt.subplots(figsize=(6, 6))
    _draw_map_background(ax, meta)
    ax.plot(xs, ys, linewidth=1.5, label="trajectory")
    ax.scatter([xs[0]], [ys[0]], color="blue", label="start", zorder=3)
    ax.scatter([xs[-1]], [ys[-1]], color="red", label="end", zorder=3)
    if "x" in dest and "y" in dest:
        ax.scatter([dest["x"]], [dest["y"]], color="green", label="goal", zorder=3)
    ax.set_title(f"Trajectory: {_method_label(meta)}")
    ax.set_xlabel("map x (m)")
    ax.set_ylabel("map y (m)")
    ax.axis("equal")
    ax.legend()
    fig.tight_layout()
    out = output_dir / f"trajectory_exp_{meta['experiment_id']}_flow_{meta['flow']}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _trajectory_xy(run_dir: Path) -> tuple[list[float], list[float]]:
    pose_rows = _load_jsonl(run_dir / "pose_map_stream.jsonl")
    xs = [float(row["x"]) for row in pose_rows if "x" in row]
    ys = [float(row["y"]) for row in pose_rows if "y" in row]
    return xs, ys


def _plot_experiment_trajectory_overlay(metadata_rows: list[dict], experiment_id: int, output: Path) -> None:
    rows = [meta for meta in metadata_rows if int(meta["experiment_id"]) == experiment_id]
    if not rows:
        return

    fig, ax = plt.subplots(figsize=(8, 8))
    _draw_map_background(ax, rows[0])
    plotted_any = False

    for meta in sorted(rows, key=lambda item: int(item["flow"])):
        run_dir = Path(meta["_run_dir"])
        xs, ys = _trajectory_xy(run_dir)
        if not xs or not ys:
            continue
        flow = str(meta["flow"])
        color = FLOW_COLORS.get(flow, None)
        ax.plot(xs, ys, linewidth=2.0, color=color, label=_method_label(meta))
        ax.scatter([xs[0]], [ys[0]], color=color, s=28, zorder=3)
        ax.scatter([xs[-1]], [ys[-1]], color=color, marker="x", s=50, zorder=3)
        plotted_any = True

    if not plotted_any:
        plt.close(fig)
        return

    dest = rows[0].get("destination_map", {})
    if "x" in dest and "y" in dest:
        ax.scatter([dest["x"]], [dest["y"]], color="black", marker="*", s=140, label="goal", zorder=4)

    ax.set_title(f"Experiment {experiment_id} Trajectories by Method")
    ax.set_xlabel("map x (m)")
    ax.set_ylabel("map y (m)")
    ax.axis("equal")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def _plot_trajectory_grid(metadata_rows: list[dict], output: Path, title: str) -> None:
    rows = [meta for meta in metadata_rows if _trajectory_xy(Path(meta["_run_dir"]))[0]]
    if not rows:
        return

    cols = 2
    n = len(rows)
    rows_count = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows_count, cols, figsize=(8 * cols, 6 * rows_count))
    if not isinstance(axes, (list, tuple)):
        try:
            axes_flat = axes.ravel()
        except AttributeError:
            axes_flat = [axes]
    else:
        axes_flat = axes

    for ax, meta in zip(axes_flat, rows):
        run_dir = Path(meta["_run_dir"])
        xs, ys = _trajectory_xy(run_dir)
        _draw_map_background(ax, meta)
        color = FLOW_COLORS.get(str(meta["flow"]), None)
        ax.plot(xs, ys, linewidth=1.8, color=color)
        ax.scatter([xs[0]], [ys[0]], color=color, s=24, zorder=3)
        ax.scatter([xs[-1]], [ys[-1]], color=color, marker="x", s=44, zorder=3)
        dest = meta.get("destination_map", {})
        if "x" in dest and "y" in dest:
            ax.scatter([dest["x"]], [dest["y"]], color="black", marker="*", s=120, zorder=4)
        ax.set_title(f"Exp {meta['experiment_id']} {_method_label(meta)}")
        ax.set_xlabel("map x (m)")
        ax.set_ylabel("map y (m)")
        ax.axis("equal")

    for ax in axes_flat[n:]:
        ax.axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def _plot_distance_vs_runtime(metadata_rows: list[dict], output: Path) -> None:
    if not metadata_rows:
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    for meta in metadata_rows:
        runtime = float(meta.get("wall_clock_sec", 0.0))
        path_length = float(meta.get("_path_length_m", 0.0))
        label = _run_label(meta)
        ax.scatter(runtime, path_length, s=60)
        ax.annotate(label, (runtime, path_length), xytext=(5, 5), textcoords="offset points", fontsize=8)
    ax.set_title("Distance Traveled vs Runtime")
    ax.set_xlabel("runtime (sec)")
    ax.set_ylabel("distance traveled (m)")
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def _plot_path_efficiency(labels: list[str], values: list[float], output: Path) -> None:
    _plot_bar(
        labels,
        values,
        "Path Efficiency by Run",
        "straight-line / traveled distance",
        output,
    )


def _cleanup_old_outputs(output_root: Path) -> None:
    if output_root.exists():
        shutil.rmtree(output_root)


def _path_efficiency(meta: dict) -> float:
    path_length = float(meta.get("_path_length_m", 0.0))
    if path_length <= 0.0:
        return 0.0
    straight_line = _straight_line_goal_distance_m(meta)
    return straight_line / path_length


def _infer_source_from_pose(run_dir: Path, meta: dict) -> None:
    if "source_map" in meta:
        return
    pose_rows = _load_jsonl(run_dir / "pose_map_stream.jsonl")
    for row in pose_rows:
        if "x" in row and "y" in row:
            meta["source_map"] = {"x": float(row["x"]), "y": float(row["y"])}
            return


def _write_experiment_summary(metadata_rows: list[dict], output: Path) -> None:
    summary = {
        "run_count": len(metadata_rows),
        "runs": [
            {
                "label": _run_label(meta),
                "method": _method_label(meta),
                "run_dir": meta["_run_dir"],
                "outcome": meta.get("outcome"),
                "flow": meta.get("flow"),
                "experiment_id": meta.get("experiment_id"),
            }
            for meta in metadata_rows
        ],
    }
    output.write_text(json.dumps(summary, indent=2))


def _select_latest_goal_reached_runs(
    metadata_rows: list[dict],
    experiment_id: int,
    flows: list[int],
) -> list[dict]:
    selected: list[dict] = []
    for flow in flows:
        candidates = [
            meta
            for meta in metadata_rows
            if int(meta["experiment_id"]) == experiment_id
            and int(meta["flow"]) == flow
            and str(meta.get("outcome", "")).lower() == "goal_reached"
        ]
        if not candidates:
            continue
        candidates.sort(key=lambda item: str(item["_run_dir"]))
        selected.append(candidates[-1])
    return selected


def _write_csv_table(rows: list[dict], output: Path) -> None:
    if not rows:
        return

    fieldnames = [
        "experiment_id",
        "method",
        "flow",
        "run_dir",
        "outcome",
        "wall_clock_sec",
        "path_length_m",
        "path_efficiency_ratio",
        "final_distance_to_goal_m",
        "total_llm_cycles",
        "total_actions",
        "total_pose_samples_map",
        "total_cmd_vel_samples",
        "llm_provider",
        "llm_model",
    ]
    with output.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for meta in rows:
            writer.writerow(
                {
                    "experiment_id": int(meta["experiment_id"]),
                    "method": _method_label(meta),
                    "flow": int(meta["flow"]),
                    "run_dir": meta["_run_dir"],
                    "outcome": meta.get("outcome"),
                    "wall_clock_sec": float(meta.get("wall_clock_sec", 0.0)),
                    "path_length_m": float(meta.get("_path_length_m", 0.0)),
                    "path_efficiency_ratio": float(meta.get("_path_efficiency", 0.0)),
                    "final_distance_to_goal_m": float(meta.get("final_distance_to_goal_m", 0.0)),
                    "total_llm_cycles": int(meta.get("total_llm_cycles", 0) or 0),
                    "total_actions": int(meta.get("total_actions", 0) or 0),
                    "total_pose_samples_map": int(meta.get("total_pose_samples_map", 0) or 0),
                    "total_cmd_vel_samples": int(meta.get("total_cmd_vel_samples", 0) or 0),
                    "llm_provider": meta.get("llm_provider", ""),
                    "llm_model": meta.get("llm_model", ""),
                }
            )


def _write_method_comparison_outputs(
    metadata_rows: list[dict],
    experiment_id: int,
    output_root: Path,
) -> None:
    selected = _select_latest_goal_reached_runs(metadata_rows, experiment_id, [5, 6, 7])
    if not selected:
        return

    comparison_dir = output_root / f"experiment_{experiment_id}" / "working_methods_5_7"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    labels = [_method_label(meta) for meta in selected]
    runtimes = [float(meta.get("wall_clock_sec", 0.0)) for meta in selected]
    path_lengths = [float(meta.get("_path_length_m", 0.0)) for meta in selected]
    final_distances = [float(meta.get("final_distance_to_goal_m", 0.0)) for meta in selected]
    llm_cycles = [float(meta.get("total_llm_cycles", 0.0)) for meta in selected]
    path_efficiencies = [float(meta.get("_path_efficiency", 0.0)) for meta in selected]

    _plot_bar(
        labels,
        runtimes,
        f"Experiment {experiment_id} Runtime by Method",
        "Runtime (s)",
        comparison_dir / "runtime_by_method.png",
    )
    _plot_bar(
        labels,
        path_lengths,
        f"Experiment {experiment_id} Distance Traveled by Method",
        "Distance traveled (m)",
        comparison_dir / "distance_traveled_by_method.png",
    )
    _plot_bar(
        labels,
        final_distances,
        f"Experiment {experiment_id} Final Distance to Goal by Method",
        "Final distance to goal (m)",
        comparison_dir / "final_distance_to_goal_by_method.png",
    )
    _plot_bar(
        labels,
        llm_cycles,
        f"Experiment {experiment_id} LLM Cycles by Method",
        "LLM cycles (count)",
        comparison_dir / "llm_cycles_by_method.png",
    )
    _plot_bar(
        labels,
        path_efficiencies,
        f"Experiment {experiment_id} Path Efficiency by Method",
        "Path efficiency (straight-line distance / traveled distance)",
        comparison_dir / "path_efficiency_by_method.png",
    )
    _plot_experiment_trajectory_overlay(
        selected,
        experiment_id,
        comparison_dir / "trajectories_overlay_by_method.png",
    )
    _plot_trajectory_grid(
        selected,
        comparison_dir / "trajectory_grid_by_method.png",
        f"Experiment {experiment_id} Working Methods Trajectory Maps",
    )
    _write_csv_table(selected, comparison_dir / "working_methods_5_7_summary.csv")
    _write_experiment_summary(selected, comparison_dir / "summary.json")


def _write_experiment_outputs(metadata_rows: list[dict], output_root: Path) -> None:
    if not metadata_rows:
        return

    experiment_id = int(metadata_rows[0]["experiment_id"])
    experiment_dir = output_root / f"experiment_{experiment_id}"
    experiment_dir.mkdir(parents=True, exist_ok=True)

    for meta in metadata_rows:
        _plot_trajectory(Path(meta["_run_dir"]), meta, experiment_dir)

    labels = [_run_label(meta) for meta in metadata_rows]
    runtimes = [float(meta.get("wall_clock_sec", 0.0)) for meta in metadata_rows]
    cycles = [float(meta.get("total_llm_cycles", 0.0)) for meta in metadata_rows]
    path_lengths = [float(meta.get("_path_length_m", 0.0)) for meta in metadata_rows]
    path_efficiencies = [float(meta.get("_path_efficiency", 0.0)) for meta in metadata_rows]

    _plot_bar(
        labels,
        runtimes,
        f"Experiment {experiment_id} Runtime by Run",
        "wall clock seconds",
        experiment_dir / "runtime_by_run.png",
    )
    _plot_bar(
        labels,
        cycles,
        f"Experiment {experiment_id} LLM Cycles by Run",
        "cycles",
        experiment_dir / "llm_cycles_by_run.png",
    )
    _plot_bar(
        labels,
        path_lengths,
        f"Experiment {experiment_id} Distance Traveled by Run",
        "meters",
        experiment_dir / "distance_traveled_by_run.png",
    )
    _plot_distance_vs_runtime(metadata_rows, experiment_dir / "distance_vs_runtime.png")
    _plot_path_efficiency(labels, path_efficiencies, experiment_dir / "path_efficiency_by_run.png")
    _plot_experiment_trajectory_overlay(
        metadata_rows,
        experiment_id,
        experiment_dir / "trajectories_overlay.png",
    )
    _plot_trajectory_grid(
        metadata_rows,
        experiment_dir / "trajectory_grid.png",
        f"Experiment {experiment_id} Trajectory Maps",
    )
    _write_experiment_summary(metadata_rows, experiment_dir / "summary.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="experiment_data_folder")
    parser.add_argument("--output", default="output")
    args = parser.parse_args()

    input_arg = Path(args.input)
    output_arg = Path(args.output)

    # Resolve defaults relative to the repo/script so the command works from
    # either the repo root or inside graphs/.
    if input_arg.is_absolute():
        input_root = input_arg
    else:
        input_root = (REPO_ROOT / input_arg).resolve()

    if output_arg.is_absolute():
        output_root = output_arg
    else:
        output_root = (SCRIPT_DIR / output_arg).resolve()

    output_root.mkdir(parents=True, exist_ok=True)
    _cleanup_old_outputs(output_root)

    run_dirs = _find_run_dirs(input_root)
    if not run_dirs:
        raise SystemExit(f"No experiment runs found under {input_root}")

    metadata_rows: list[dict] = []

    for run_dir in run_dirs:
        meta = _load_json(run_dir / "metadata.json")
        meta["_run_dir"] = str(run_dir)
        meta["_path_length_m"] = _path_length_m(run_dir)
        _infer_source_from_pose(run_dir, meta)
        meta["_path_efficiency"] = _path_efficiency(meta)
        metadata_rows.append(meta)

    experiment_ids = sorted({int(meta["experiment_id"]) for meta in metadata_rows})
    for experiment_id in experiment_ids:
        experiment_rows = [
            meta for meta in metadata_rows if int(meta["experiment_id"]) == experiment_id
        ]
        _write_experiment_outputs(experiment_rows, output_root)
        _write_method_comparison_outputs(metadata_rows, experiment_id, output_root)


if __name__ == "__main__":
    main()
