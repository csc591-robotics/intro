#!/usr/bin/env python3
"""Read custom_map_builder/maps/map_poses.yaml + the per-world sidecar and
emit shell-evaluable variables for a given map name.

Usage inside a shell script:

    eval "$(python3 scripts/parse_map_poses.py MAP_NAME [MAP_POSES_PATH])"

MAP_NAME is the key inside ``map_poses.yaml`` (e.g. ``warehouse.pgm`` or just
``warehouse``).  The script accepts the bare stem too and resolves
``<stem>.pgm`` automatically.

The variables printed include both the Gazebo-world-frame coordinates (used to
spawn the robot in Gazebo via spawn_entity.py) and the map-frame coordinates
(used by the LLM agent which compares pose lookups against ``map -> base_link``
TF).  The relationship between them is:

    map_xy = world_xy + sidecar.world_to_map_offset

For hand-crafted maps with no sidecar, the offset is (0, 0) so map and world
frames are identical and the YAML is interpreted as map-frame already.
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import yaml


def _quat_to_yaw(qx: float, qy: float, qz: float, qw: float) -> float:
    siny = 2.0 * (qw * qz + qx * qy)
    cosy = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny, cosy)


def _resolve_workspace_root(start: Path) -> Path:
    """Walk up from ``start`` to find the colcon workspace (contains src/)."""
    cur = start.resolve()
    for _ in range(8):
        if (cur / 'src').is_dir():
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    return start.resolve()


def _resolve_map_poses(explicit: str | None) -> Path:
    if explicit:
        p = Path(explicit).expanduser().resolve()
        if not p.is_file():
            print(f'map_poses.yaml not found: {p}', file=sys.stderr)
            sys.exit(1)
        return p

    here = Path(__file__).resolve()
    workspace = _resolve_workspace_root(here)
    candidate = workspace / 'src' / 'custom_map_builder' / 'maps' / 'map_poses.yaml'
    if candidate.is_file():
        return candidate

    print(
        'Could not locate map_poses.yaml. Pass it explicitly as the second '
        'argument.', file=sys.stderr,
    )
    sys.exit(1)


def _resolve_sidecar(sidecar_field: str, workspace: Path) -> Path | None:
    if not sidecar_field:
        return None
    p = Path(sidecar_field)
    if not p.is_absolute():
        p = workspace / p
    p = p.resolve()
    if not p.is_file():
        return None
    return p


def _shquote(value: str) -> str:
    return "'" + value.replace("'", "'\\''") + "'"


def main() -> None:
    if len(sys.argv) < 2:
        print('Usage: parse_map_poses.py MAP_NAME [MAP_POSES_PATH]',
              file=sys.stderr)
        sys.exit(1)

    map_name_in = sys.argv[1].strip()
    map_poses_path = _resolve_map_poses(
        sys.argv[2] if len(sys.argv) >= 3 else None
    )

    workspace = _resolve_workspace_root(map_poses_path.parent)

    with map_poses_path.open() as fh:
        data = yaml.safe_load(fh) or {}

    maps = data.get('maps') or {}

    candidates = [
        map_name_in,
        map_name_in if map_name_in.endswith('.pgm') else map_name_in + '.pgm',
        map_name_in[:-4] if map_name_in.endswith('.pgm') else map_name_in,
    ]
    entry = None
    map_key = None
    for cand in candidates:
        if cand in maps:
            entry = maps[cand]
            map_key = cand
            break
    if entry is None:
        avail = ', '.join(sorted(maps.keys())) if maps else '(none)'
        print(f'Map {map_name_in!r} not found in {map_poses_path}. '
              f'Available: {avail}', file=sys.stderr)
        sys.exit(1)

    sidecar_path = _resolve_sidecar(entry.get('sidecar') or '', workspace)

    sidecar = {}
    map_yaml_path = ''
    world_file = ''
    map_offset_x = 0.0
    map_offset_y = 0.0

    if sidecar_path is not None:
        with sidecar_path.open() as fh:
            sidecar = yaml.safe_load(fh) or {}
        offset = sidecar.get('world_to_map_offset') or [0.0, 0.0]
        map_offset_x = float(offset[0])
        map_offset_y = float(offset[1])
        recorded_world = sidecar.get('source_world') or ''
        if recorded_world and os.path.isfile(recorded_world):
            world_file = recorded_world
        elif recorded_world:
            world_basename = os.path.basename(recorded_world)
            world_search_dirs = [
                workspace / 'world_files',
                workspace / 'src' / 'world_to_map' / 'world_files',
                workspace / 'world_files'
                    / 'gazebo_models_worlds_collection' / 'worlds',
                workspace / 'src' / 'world_to_map' / 'world_files'
                    / 'gazebo_models_worlds_collection' / 'worlds',
            ]
            for d in world_search_dirs:
                cand = d / world_basename
                if cand.is_file():
                    world_file = str(cand.resolve())
                    break
        map_yaml_field = sidecar.get('map_yaml') or ''
        if map_yaml_field:
            mp = Path(map_yaml_field)
            if not mp.is_absolute():
                mp = sidecar_path.parent / mp
            map_yaml_path = str(mp.resolve())

    if not map_yaml_path:
        stem = map_key[:-4] if map_key.endswith('.pgm') else map_key
        legacy = workspace / 'src' / 'nav2_llm_demo' / 'maps' / f'{stem}.yaml'
        if legacy.is_file():
            map_yaml_path = str(legacy.resolve())
        else:
            cmb = workspace / 'src' / 'custom_map_builder' / 'maps' / f'{stem}.yaml'
            if cmb.is_file():
                map_yaml_path = str(cmb.resolve())

    src = entry.get('source') or {}
    dst = entry.get('destination') or {}

    src_pos = src.get('position', {}) if isinstance(src, dict) else {}
    src_ori = src.get('orientation', {}) if isinstance(src, dict) else {}
    dst_pos = dst.get('position', {}) if isinstance(dst, dict) else {}
    dst_ori = dst.get('orientation', {}) if isinstance(dst, dict) else {}

    world_sx = float(src_pos.get('x', 0.0))
    world_sy = float(src_pos.get('y', 0.0))
    world_sz = max(float(src_pos.get('z', 0.05)), 0.05)
    world_dx = float(dst_pos.get('x', world_sx))
    world_dy = float(dst_pos.get('y', world_sy))

    src_yaw = _quat_to_yaw(
        float(src_ori.get('x', 0.0)), float(src_ori.get('y', 0.0)),
        float(src_ori.get('z', 0.0)), float(src_ori.get('w', 1.0)),
    ) if isinstance(src, dict) and src_ori else float(src.get('yaw_rad', 0.0)) if isinstance(src, dict) else 0.0
    dst_yaw = _quat_to_yaw(
        float(dst_ori.get('x', 0.0)), float(dst_ori.get('y', 0.0)),
        float(dst_ori.get('z', 0.0)), float(dst_ori.get('w', 1.0)),
    ) if isinstance(dst, dict) and dst_ori else 0.0

    map_sx = world_sx + map_offset_x
    map_sy = world_sy + map_offset_y
    map_dx = world_dx + map_offset_x
    map_dy = world_dy + map_offset_y

    # The TurtleBot3 gazebo diff-drive plugin (odometry_source=WORLD) publishes
    # ``odom -> base_footprint`` matching the robot's absolute world pose, so
    # ``odom`` effectively equals the Gazebo world frame.  The static
    # ``map -> odom`` TF therefore only needs the sidecar offset; adding the
    # spawn pose on top would double-count and push base_footprint outside the
    # map.
    static_tf_x = map_offset_x
    static_tf_y = map_offset_y

    gazebo_model_path = ''
    candidate_models = [
        workspace / 'world_files' / 'gazebo_models_worlds_collection' / 'models',
        workspace / 'src' / 'world_to_map' / 'world_files'
            / 'gazebo_models_worlds_collection' / 'models',
    ]
    for c in candidate_models:
        if c.is_dir():
            gazebo_model_path = str(c.resolve())
            break

    sidecar_str = str(sidecar_path) if sidecar_path else ''
    has_world = '1' if world_file and os.path.isfile(world_file) else '0'

    lines = [
        f'MAP_NAME={_shquote(map_key)}',
        f'MAP_POSES_PATH={_shquote(str(map_poses_path))}',
        f'MAP_YAML_PATH={_shquote(map_yaml_path)}',
        f'SIDECAR_PATH={_shquote(sidecar_str)}',
        f'WORLD_FILE={_shquote(world_file)}',
        f'HAS_WORLD={has_world}',
        f'GAZEBO_MODEL_PATH_AUTO={_shquote(gazebo_model_path)}',
        f'MAP_OFFSET_X={map_offset_x:.6f}',
        f'MAP_OFFSET_Y={map_offset_y:.6f}',
        f'SPAWN_X={world_sx:.6f}',
        f'SPAWN_Y={world_sy:.6f}',
        f'SPAWN_Z={world_sz:.6f}',
        f'SPAWN_YAW={src_yaw:.6f}',
        f'STATIC_TF_X={static_tf_x:.6f}',
        f'STATIC_TF_Y={static_tf_y:.6f}',
        f'SOURCE_X={map_sx:.6f}',
        f'SOURCE_Y={map_sy:.6f}',
        f'SOURCE_YAW={src_yaw:.6f}',
        f'DEST_X={map_dx:.6f}',
        f'DEST_Y={map_dy:.6f}',
        f'DEST_YAW={dst_yaw:.6f}',
        f'WORLD_SOURCE_X={world_sx:.6f}',
        f'WORLD_SOURCE_Y={world_sy:.6f}',
        f'WORLD_DEST_X={world_dx:.6f}',
        f'WORLD_DEST_Y={world_dy:.6f}',
    ]
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
