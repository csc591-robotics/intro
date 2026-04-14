#!/usr/bin/env python3
"""Read nav_config.yaml and emit shell-evaluable variables for a given map.

Usage inside a shell script:

    eval "$(python3 scripts/parse_nav_config.py MAP_NAME [CONFIG_PATH])"

The script prints lines like:
    MAP_YAML_PATH=/abs/path/to/diamond_blocked.yaml
    SOURCE_X=13.1001
    ...
"""

import math
import sys
from pathlib import Path

import yaml


def _quat_to_yaw(qx: float, qy: float, qz: float, qw: float) -> float:
    siny = 2.0 * (qw * qz + qx * qy)
    cosy = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny, cosy)


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: parse_nav_config.py MAP_NAME [CONFIG_PATH]", file=sys.stderr)
        sys.exit(1)

    map_name = sys.argv[1]

    if len(sys.argv) >= 3:
        config_path = Path(sys.argv[2])
    else:
        config_path = Path(__file__).resolve().parent.parent / "maps" / "nav_config.yaml"

    if not config_path.is_file():
        print(f"Config not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    with config_path.open() as f:
        config = yaml.safe_load(f)

    maps = config.get("maps", {})
    if map_name not in maps:
        available = ", ".join(sorted(maps.keys())) if maps else "(none)"
        print(f"Map '{map_name}' not found in config. Available: {available}", file=sys.stderr)
        sys.exit(1)

    entry = maps[map_name]
    maps_dir = config_path.resolve().parent

    map_yaml = entry.get("map_yaml", f"{map_name}.yaml")
    map_yaml_path = maps_dir / map_yaml

    src = entry.get("source") or {}
    dst = entry.get("destination") or {}

    src_pos = src.get("position", {})
    src_ori = src.get("orientation", {})
    dst_pos = dst.get("position", {})
    dst_ori = dst.get("orientation", {})

    src_yaw = _quat_to_yaw(
        src_ori.get("x", 0), src_ori.get("y", 0),
        src_ori.get("z", 0), src_ori.get("w", 1),
    )
    dst_yaw = _quat_to_yaw(
        dst_ori.get("x", 0), dst_ori.get("y", 0),
        dst_ori.get("z", 0), dst_ori.get("w", 1),
    )

    lines = [
        f'MAP_YAML_PATH="{map_yaml_path}"',
        f'MAP_PGM_DIR="{maps_dir}"',
        f'SOURCE_X={src_pos.get("x", 0.0)}',
        f'SOURCE_Y={src_pos.get("y", 0.0)}',
        f'SOURCE_Z={src_pos.get("z", 0.0)}',
        f'SOURCE_QX={src_ori.get("x", 0.0)}',
        f'SOURCE_QY={src_ori.get("y", 0.0)}',
        f'SOURCE_QZ={src_ori.get("z", 0.0)}',
        f'SOURCE_QW={src_ori.get("w", 1.0)}',
        f'SOURCE_YAW={src_yaw}',
        f'DEST_X={dst_pos.get("x", 0.0)}',
        f'DEST_Y={dst_pos.get("y", 0.0)}',
        f'DEST_Z={dst_pos.get("z", 0.0)}',
        f'DEST_QX={dst_ori.get("x", 0.0)}',
        f'DEST_QY={dst_ori.get("y", 0.0)}',
        f'DEST_QZ={dst_ori.get("z", 0.0)}',
        f'DEST_QW={dst_ori.get("w", 1.0)}',
        f'DEST_YAW={dst_yaw}',
    ]
    print("\n".join(lines))


if __name__ == "__main__":
    main()
