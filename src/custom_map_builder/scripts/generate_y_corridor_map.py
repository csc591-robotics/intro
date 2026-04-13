#!/usr/bin/env python3
"""Generate a simple Y-shaped free-space map (PGM + YAML) for testing map_builder.

Run from anywhere; default output is next to this script under ../maps/.
"""
from __future__ import annotations

import argparse
from pathlib import Path


def build_grid(width: int, height: int) -> list[list[int]]:
    """Return grayscale rows (top row first), 255=free, 0=occupied."""
    grid = [[255 for _ in range(width)] for _ in range(height)]

    def fill_rect(x0: int, y0: int, x1: int, y1: int, value: int) -> None:
        for y in range(max(0, y0), min(height, y1 + 1)):
            for x in range(max(0, x0), min(width, x1 + 1)):
                grid[y][x] = value

    # Outer walls
    fill_rect(0, 0, width - 1, 0, 0)
    fill_rect(0, height - 1, width - 1, height - 1, 0)
    fill_rect(0, 0, 0, height - 1, 0)
    fill_rect(width - 1, 0, width - 1, height - 1, 0)

    cx, cy = width // 2, height // 2
    t = 3  # half-thickness of walls

    # Vertical stem (bottom to center)
    fill_rect(cx - t, cy, cx + t, height - 2, 0)
    # Left branch
    fill_rect(2, cy - t, cx, cy + t, 0)
    # Right branch
    fill_rect(cx, cy - t, width - 3, cy + t, 0)
    # Upper fork walls (opening at top center)
    fill_rect(cx - t, 2, cx + t, cy - t, 0)

    # Carve free corridors (overwrite walls with free)
    fill_rect(cx - t + 1, cy + 1, cx + t - 1, height - 2, 255)  # stem channel
    fill_rect(3, cy - t + 1, cx - t - 1, cy + t - 1, 255)  # left arm
    fill_rect(cx + t + 1, cy - t + 1, width - 4, cy + t - 1, 255)  # right arm
    fill_rect(cx - t + 1, 3, cx + t - 1, cy - t - 1, 255)  # upper stem to fork

    return grid


def write_pgm_p2(path: Path, grid: list[list[int]]) -> None:
    height = len(grid)
    width = len(grid[0])
    lines = ['P2', f'{width} {height}', '255']
    for row in grid:
        lines.append(' '.join(str(v) for v in row))
    path.write_text('\n'.join(lines) + '\n', encoding='ascii')


def write_yaml(path: Path, pgm_name: str, resolution: float, ox: float, oy: float) -> None:
    text = f"""image: {pgm_name}
resolution: {resolution}
origin: [{ox}, {oy}, 0.0]
negate: 0
occupied_thresh: 0.65
free_thresh: 0.25
mode: trinary
"""
    path.write_text(text, encoding='utf-8')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--width', type=int, default=120)
    parser.add_argument('--height', type=int, default=120)
    parser.add_argument('--resolution', type=float, default=0.05)
    parser.add_argument('--origin-x', type=float, default=-3.0)
    parser.add_argument('--origin-y', type=float, default=-3.0)
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path(__file__).resolve().parent.parent / 'maps',
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pgm_path = args.output_dir / 'y_corridor.pgm'
    yaml_path = args.output_dir / 'y_corridor.yaml'
    grid = build_grid(args.width, args.height)
    write_pgm_p2(pgm_path, grid)
    write_yaml(yaml_path, pgm_path.name, args.resolution, args.origin_x, args.origin_y)
    print(f'Wrote {pgm_path} and {yaml_path}')


if __name__ == '__main__':
    main()
