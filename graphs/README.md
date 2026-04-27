# Graphs

Simple matplotlib utilities for plotting experiment outputs from
`experiment_data_folder/`.

## Usage

From the repo root:

```bash
python3 graphs/plot_experiments.py
```

This reads the latest data under `experiment_data_folder/` and writes PNGs to:

```bash
graphs/output/
```

Inside that folder, outputs are split by experiment:

```bash
graphs/output/experiment_1/
graphs/output/experiment_2/
```

## Optional flags

```bash
python3 graphs/plot_experiments.py \
  --input experiment_data_folder \
  --output graphs/output
```

## What it makes

- `experiment_<id>/distance_traveled_by_run.png`
- `experiment_<id>/runtime_by_run.png`
- `experiment_<id>/trajectory_exp_<experiment>_flow_<flow>.png`
- `experiment_<id>/trajectories_overlay.png`
- `experiment_<id>/trajectory_grid.png`
- `experiment_<id>/llm_cycles_by_run.png`
- `experiment_<id>/distance_vs_runtime.png`
- `experiment_<id>/path_efficiency_by_run.png`
