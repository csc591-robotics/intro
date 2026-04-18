# `world_to_map`

Rasterize a Gazebo `.world` file into a Nav2-style **PGM + YAML** occupancy
map and launch **Gazebo + map_server + RViz + TurtleBot3 Burger** so the
robot's pose is shown 1:1 in Gazebo and RViz at the same metric scale
(2 m driven in Gazebo == 2 m on the RViz map).

Drive the robot with the standard TurtleBot3 keyboard teleop
(`w` / `x` forward-back, `a` / `d` turn, `s` stop) in a second terminal.

## Where worlds and models live

The runner searches for worlds in two places, in order:

1. `intro/world_files/<name>.world` — original 4 worlds (e.g. `diamond_map`).
2. `intro/world_files/gazebo_models_worlds_collection/worlds/<name>.world`
   — the bundled collection of ~50 community worlds.

Models for the collection live at:

- `intro/world_files/gazebo_models_worlds_collection/models/`

The runner automatically prepends that directory to `GAZEBO_MODEL_PATH`,
so Gazebo can resolve every `model://...` include and the rasterizer can
walk into each model's `model.sdf` to extract its collision boxes /
cylinders. List every world the runner knows about:

```bash
bash intro/src/world_to_map/run_world_to_map.sh
```

## Quick start (inside the Docker container)

```bash
cd /workspace/intro
colcon build --packages-select world_to_map
source install/setup.bash
bash src/world_to_map/run_world_to_map.sh workshop_example
```

In a **second** terminal in the same container, drive with the bundled
WASDX teleop (no extra apt packages needed):

```bash
docker compose exec autonomous_pathing_llm bash
export TURTLEBOT3_MODEL=burger
source /opt/ros/humble/setup.bash && source install/setup.bash
ros2 run world_to_map teleop_wasdx
```

Controls: `w` / `x` forward/back, `a` / `d` turn, `s` stop, `q` quit,
`+` / `-` change linear step, `]` / `[` change angular step.

## What the runner does

1. Resolves the world name to a `.world` path (primary folder first, then
   the collection).
2. Sets `GAZEBO_MODEL_PATH` to include the collection's `models/`
   directory (plus anything in `EXTRA_GAZEBO_MODEL_PATH`).
3. Runs the rasterizer to produce three sibling files in
   `intro/src/world_to_map/maps/`:
   - `<name>.pgm` — the occupancy grid (free / occupied only)
   - `<name>.yaml` — Nav2 `map_server` config
   - `<name>.world_map.yaml` — sidecar that documents the world ↔ map
     transform (resolution, origin, padded AABB, frames, world->map offset)
     so any other package can reuse it.
4. `ros2 launch world_to_map world_to_map.launch.py` — starts `gzserver`,
   `gzclient`, spawns the TurtleBot3, runs `map_server` + lifecycle
   manager, publishes the static `map -> odom` transform with the offset
   from the sidecar, and opens RViz.

## Origin modes

`ORIGIN_MODE` (default `bottom-left`) controls where the bottom-left
pixel of the PGM lives in the **map** frame:

- `bottom-left` — `map.yaml`'s `origin = [0, 0, 0]`. The bottom-left
  corner of the rasterized image is at the **map frame origin**, so the
  RViz map sits in the positive quadrant. The launch then publishes a
  static TF `map -> odom` with translation
  `(-padded_min_x, -padded_min_y)` so the robot's `(x, y, yaw)` from
  Gazebo still moves 1:1 across both views.
- `world` — `map.yaml`'s `origin = [padded_min_x, padded_min_y, 0]`.
  The **map frame is identical to Gazebo's world frame**; the static TF
  is identity. Use this when you want one-to-one frames at the cost of
  the map straddling negative coordinates.

The rasterizer always preserves the metric scale: `resolution` (default
`0.05` m/px) is meters per pixel and the image is sized exactly
`ceil((max - min) / resolution)`. There is no rescaling, normalisation,
or fit-to-image step.

## Reusable scaling sidecar (`<name>.world_map.yaml`)

Excerpt:

```yaml
schema_version: 2
generated_by: world_to_map.rasterize_world
source_world: /workspace/intro/world_files/gazebo_models_worlds_collection/worlds/workshop_example.world
map_yaml: workshop_example.yaml
map_pgm: workshop_example.pgm
resolution: 0.05
origin: [0.0, 0.0, 0.0]
image_size_px: [180, 160]
world_extent_m: [9.0, 8.0]
world_aabb:
  min: [-3.5, -3.0]
  max: [3.5, 3.0]
padded_aabb:
  min: [-4.5, -4.0]
  max: [4.5, 4.0]
padding_m: 1.0
z_band_m: [0.0, 0.4]
origin_mode: bottom-left
world_to_map_offset: [4.5, 4.0]    # = -padded_min
frames:
  map_frame: map
  world_frame: world
  map_to_odom_translation: [4.5, 4.0, 0.0]
model_paths:
  - /workspace/intro/world_files/gazebo_models_worlds_collection/models
stats:
  boxes: 312
  cylinders: 4
  includes_resolved: 28
  includes_unresolved: []
  skipped_mesh: 14
  skipped_other: 0
```

Other packages convert between Gazebo `world` coordinates and map pixels
without re-parsing anything:

```python
import yaml, pathlib
meta = yaml.safe_load(pathlib.Path('workshop_example.world_map.yaml').read_text())
res = meta['resolution']
ox, oy, _ = meta['origin']               # bottom-left of image in map frame
dx, dy = meta['world_to_map_offset']     # add to a world (x,y) to get map (x,y)

def world_to_pixel(wx, wy):
    mx, my = wx + dx, wy + dy            # world -> map
    col = int((mx - ox) / res)
    row = meta['image_size_px'][1] - 1 - int((my - oy) / res)
    return col, row
```

## CLI: rasterizer only

```bash
ros2 run world_to_map rasterize_world \
  --world /workspace/intro/world_files/gazebo_models_worlds_collection/worlds/workshop_example.world \
  --out   /workspace/intro/src/world_to_map/maps/workshop_example \
  --resolution 0.05 \
  --padding 1.0 \
  --z-min 0.0 --z-max 0.4 \
  --floor-clearance 0.05 \
  --origin-mode bottom-left \
  --model-paths /workspace/intro/world_files/gazebo_models_worlds_collection/models
```

`--floor-clearance` is what saves the workshop / warehouse worlds: many
of the SDF models in the collection bake their "floor" in as a thin
6m × 4m × 0.013m box collision sitting at z ≈ 0. Without this filter the
rasterizer stamps that slab into the PGM and the room interior becomes a
single black blob in RViz, hiding every real obstacle inside. Anything
whose top is at or below `--floor-clearance` (default 0.05m) is treated
as floor and dropped. Increase it if your world has tall ground plates;
set `0` to disable.

## Environment overrides for the runner

| Variable                  | Default        | Meaning                                                     |
|---------------------------|----------------|-------------------------------------------------------------|
| `RESOLUTION`              | `0.05`         | Meters per pixel (the "scale")                              |
| `PADDING`                 | `1.0`          | Empty border (m) around the world AABB                      |
| `Z_MIN`/`Z_MAX`           | `0.0`/`0.4`    | Robot height band used to filter obstacles                  |
| `FLOOR_CLEARANCE`         | `0.05`         | Skip box/cyl collisions whose top is `<=` this height (m)   |
| `X_POSE`/`Y_POSE`/`YAW`   | `0.0`          | TurtleBot3 spawn pose in **Gazebo world frame**             |
| `ORIGIN_MODE`             | `bottom-left`  | `bottom-left` or `world` (see above)                        |
| `LAUNCH_RVIZ`             | `true`         | Set `false` to skip RViz                                    |
| `FORCE`                   | `0`            | Set `1` to regenerate the map even if it exists             |
| `EXTRA_GAZEBO_MODEL_PATH` | (empty)        | Extra colon-separated dirs prepended to `GAZEBO_MODEL_PATH` |

## What it supports / what it skips

- Inline and included `<box>` and `<cylinder>` collision geometry —
  fully rasterized (yaw rotation supported; roll/pitch ignored).
- `<include>` of `model://...` URIs — resolved against
  `--model-paths` / `GAZEBO_MODEL_PATH`. Each model's `model.sdf` is
  parsed and its links are walked recursively.
- `<mesh>`, `<polyline>`, `<sphere>`, `<heightmap>` — skipped with a
  warning (most collision geometry in the bundled collection is
  boxes/cylinders, so this rarely matters; visuals are skipped as long
  as a `<collision>` exists for the link).
- The rasterizer prints how many includes resolved vs not, so you can
  spot model-pack misses immediately.
