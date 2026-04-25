"""Annotated map rendering for flow_5: PGM + planned path overlay.

Builds on the same PGM/origin/resolution conventions as
``map_renderer.render_annotated_map`` but draws the planned A* path on
top of everything else and crops so that the robot, the current target
waypoint, and the final destination are all in frame.
"""

from __future__ import annotations

import base64
import io
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from ..map_renderer import load_map_yaml, load_pgm, world_to_pixel


_PATH_RGB = (220, 0, 220)         # magenta
_TARGET_RGB = (0, 220, 220)       # cyan
_ROBOT_RGB = (220, 40, 40)
_ROBOT_OUTLINE = (180, 0, 0)
_SOURCE_RGB = (40, 80, 220)
_DEST_RGB = (0, 200, 0)


def _draw_arrow_from(draw: ImageDraw.ImageDraw, px: int, py: int,
                     yaw: float, length: int = 18) -> None:
    tip_x = px + int(length * math.cos(yaw))
    tip_y = py - int(length * math.sin(yaw))
    draw.line([(px, py), (tip_x, tip_y)], fill=_ROBOT_RGB, width=3)
    head_angle = 0.5
    head_len = 7
    for sign in (-1, 1):
        hx = tip_x - int(head_len * math.cos(yaw + sign * head_angle))
        hy = tip_y + int(head_len * math.sin(yaw + sign * head_angle))
        draw.line([(tip_x, tip_y), (hx, hy)], fill=_ROBOT_RGB, width=2)


def _draw_target_crosshair(draw: ImageDraw.ImageDraw, px: int, py: int,
                           radius: int = 12) -> None:
    draw.ellipse(
        [px - radius, py - radius, px + radius, py + radius],
        outline=_TARGET_RGB, width=3,
    )
    draw.line([(px - radius - 4, py), (px + radius + 4, py)],
              fill=_TARGET_RGB, width=2)
    draw.line([(px, py - radius - 4), (px, py + radius + 4)],
              fill=_TARGET_RGB, width=2)


def render_with_path(
    map_yaml_path: str | Path,
    robot_x: float,
    robot_y: float,
    robot_yaw: float,
    dest_x: float,
    dest_y: float,
    source_x: float | None,
    source_y: float | None,
    planned_path: list[tuple[float, float]],
    target_idx: int,
    crop_radius_m: float = 18.0,
    output_size: int = 512,
) -> str:
    """Render the map + planned path + markers; return base64 PNG."""
    map_yaml_path = Path(map_yaml_path)
    meta = load_map_yaml(map_yaml_path)
    pgm_filename = meta["image"]
    pgm_path = map_yaml_path.parent / pgm_filename
    resolution = float(meta["resolution"])
    origin = meta.get("origin", [0.0, 0.0, 0.0])
    origin_x, origin_y = float(origin[0]), float(origin[1])
    negate = int(meta.get("negate", 0))

    grid = load_pgm(pgm_path)
    img_h, img_w = grid.shape
    if negate:
        grid = 255 - grid

    rgb = np.zeros((img_h, img_w, 3), dtype=np.uint8)
    rgb[grid >= 200] = [255, 255, 255]
    rgb[grid <= 50] = [40, 40, 40]
    mask_unknown = (grid > 50) & (grid < 200)
    rgb[mask_unknown] = [180, 180, 180]

    img = Image.fromarray(rgb, "RGB")
    draw = ImageDraw.Draw(img)

    # Convert all relevant world points to pixels up front.
    robot_px, robot_py = world_to_pixel(
        robot_x, robot_y, origin_x, origin_y, resolution, img_h,
    )
    dest_px, dest_py = world_to_pixel(
        dest_x, dest_y, origin_x, origin_y, resolution, img_h,
    )
    src_px = src_py = None
    if source_x is not None and source_y is not None:
        src_px, src_py = world_to_pixel(
            source_x, source_y, origin_x, origin_y, resolution, img_h,
        )

    path_pixels = [
        world_to_pixel(wx, wy, origin_x, origin_y, resolution, img_h)
        for (wx, wy) in planned_path
    ]

    # Magenta polyline through all waypoints.
    if len(path_pixels) >= 2:
        draw.line(path_pixels, fill=_PATH_RGB, width=3)

    # Small magenta dots for unreached waypoints.
    safe_target_idx = max(0, min(target_idx, len(path_pixels) - 1))
    for i, (px, py) in enumerate(path_pixels):
        if i < safe_target_idx:
            continue
        if i == safe_target_idx:
            continue
        draw.ellipse(
            [px - 3, py - 3, px + 3, py + 3],
            fill=_PATH_RGB, outline=_PATH_RGB,
        )

    # Source / destination markers.
    if src_px is not None and src_py is not None:
        draw.ellipse(
            [src_px - 7, src_py - 7, src_px + 7, src_py + 7],
            outline=_SOURCE_RGB, width=2,
        )
    draw.line(
        [(dest_px - 12, dest_py), (dest_px + 12, dest_py)],
        fill=_DEST_RGB, width=3,
    )
    draw.line(
        [(dest_px, dest_py - 12), (dest_px, dest_py + 12)],
        fill=_DEST_RGB, width=3,
    )
    draw.ellipse(
        [dest_px - 12, dest_py - 12, dest_px + 12, dest_py + 12],
        outline=_DEST_RGB, width=2,
    )

    # Cyan crosshair on the current target waypoint (drawn last so it
    # sits on top of the magenta line).
    if path_pixels:
        tx_px, ty_px = path_pixels[safe_target_idx]
        _draw_target_crosshair(draw, tx_px, ty_px)
    else:
        tx_px, ty_px = robot_px, robot_py

    # Robot dot + heading arrow drawn last, on top of the path.
    draw.ellipse(
        [robot_px - 5, robot_py - 5, robot_px + 5, robot_py + 5],
        fill=_ROBOT_RGB, outline=_ROBOT_OUTLINE,
    )
    _draw_arrow_from(draw, robot_px, robot_py, robot_yaw)

    # Crop unioning robot, current target, and final destination so all
    # three are visible. Also include source for context if provided.
    crop_px = int(crop_radius_m / resolution)
    margin = max(24, int(12 / resolution))
    r0x, r0y = robot_px - crop_px, robot_py - crop_px
    r1x, r1y = robot_px + crop_px, robot_py + crop_px

    def _grow(px: int, py: int) -> None:
        nonlocal r0x, r0y, r1x, r1y
        r0x = min(r0x, px - margin)
        r0y = min(r0y, py - margin)
        r1x = max(r1x, px + margin)
        r1y = max(r1y, py + margin)

    _grow(tx_px, ty_px)
    _grow(dest_px, dest_py)
    if src_px is not None and src_py is not None:
        _grow(src_px, src_py)

    left = int(max(0, r0x))
    upper = int(max(0, r0y))
    right = int(min(img_w, r1x))
    lower = int(min(img_h, r1y))
    if left >= right or upper >= lower:
        left, upper, right, lower = 0, 0, img_w, img_h

    cropped = img.crop((left, upper, right, lower))
    cropped = cropped.resize((output_size, output_size), Image.NEAREST)

    buf = io.BytesIO()
    cropped.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


__all__ = ["render_with_path"]
