"""Render an annotated top-down map image for the vision LLM.

Loads the PGM occupancy grid, draws the robot position/heading, the
destination marker, and the source marker, then returns a base64-encoded
PNG suitable for multimodal LLM messages.
"""

import base64
import io
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


def load_map_yaml(yaml_path: str | Path) -> dict[str, Any]:
    """Parse a Nav2 map YAML without requiring the full yaml library at import."""
    import yaml

    with open(yaml_path) as f:
        return yaml.safe_load(f)


def load_pgm(pgm_path: str | Path) -> np.ndarray:
    """Load a PGM file as a numpy array (grayscale, 0-255)."""
    img = Image.open(pgm_path).convert("L")
    return np.array(img)


def world_to_pixel(
    wx: float,
    wy: float,
    origin_x: float,
    origin_y: float,
    resolution: float,
    img_height: int,
) -> tuple[int, int]:
    """Convert world coordinates (meters) to pixel coordinates."""
    px = int((wx - origin_x) / resolution)
    py = img_height - 1 - int((wy - origin_y) / resolution)
    return px, py


def render_annotated_map(
    map_yaml_path: str | Path,
    robot_x: float,
    robot_y: float,
    robot_yaw: float,
    dest_x: float,
    dest_y: float,
    source_x: float | None = None,
    source_y: float | None = None,
    crop_radius_m: float = 8.0,
    output_size: int = 512,
) -> str:
    """Render the map with robot and destination markers, return base64 PNG.

    Parameters
    ----------
    map_yaml_path:
        Path to the Nav2 map YAML file.
    robot_x, robot_y, robot_yaw:
        Current robot pose in the map frame (meters, radians).
    dest_x, dest_y:
        Destination position in the map frame.
    source_x, source_y:
        Optional source/start position to mark.
    crop_radius_m:
        Radius in meters around the robot to crop. Set large to show full map.
    output_size:
        Final image dimension (square) in pixels.

    Returns
    -------
    Base64-encoded PNG string.
    """
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
    rgb[grid >= 200] = [255, 255, 255]  # free = white
    rgb[grid <= 50] = [40, 40, 40]       # occupied = dark gray
    mask_unknown = (grid > 50) & (grid < 200)
    rgb[mask_unknown] = [180, 180, 180]  # unknown = light gray

    img = Image.fromarray(rgb, "RGB")
    draw = ImageDraw.Draw(img)

    robot_px, robot_py = world_to_pixel(
        robot_x, robot_y, origin_x, origin_y, resolution, img_h,
    )
    dest_px, dest_py = world_to_pixel(
        dest_x, dest_y, origin_x, origin_y, resolution, img_h,
    )

    _draw_destination(draw, dest_px, dest_py, radius=12)

    if source_x is not None and source_y is not None:
        src_px, src_py = world_to_pixel(
            source_x, source_y, origin_x, origin_y, resolution, img_h,
        )
        _draw_source(draw, src_px, src_py, radius=8)

    _draw_robot(draw, robot_px, robot_py, robot_yaw, arrow_len=18, img_height=img_h)

    crop_px = int(crop_radius_m / resolution)
    left = max(0, robot_px - crop_px)
    upper = max(0, robot_py - crop_px)
    right = min(img_w, robot_px + crop_px)
    lower = min(img_h, robot_py + crop_px)

    if left >= right or upper >= lower:
        left, upper, right, lower = 0, 0, img_w, img_h

    cropped = img.crop((left, upper, right, lower))
    cropped = cropped.resize((output_size, output_size), Image.NEAREST)

    buf = io.BytesIO()
    cropped.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def render_full_map(
    map_yaml_path: str | Path,
    robot_x: float,
    robot_y: float,
    robot_yaw: float,
    dest_x: float,
    dest_y: float,
    source_x: float | None = None,
    source_y: float | None = None,
    output_size: int = 768,
) -> str:
    """Render the full map (no crop) with markers, return base64 PNG."""
    return render_annotated_map(
        map_yaml_path=map_yaml_path,
        robot_x=robot_x,
        robot_y=robot_y,
        robot_yaw=robot_yaw,
        dest_x=dest_x,
        dest_y=dest_y,
        source_x=source_x,
        source_y=source_y,
        crop_radius_m=99999.0,
        output_size=output_size,
    )


def _draw_robot(
    draw: ImageDraw.ImageDraw,
    px: int,
    py: int,
    yaw: float,
    arrow_len: int = 18,
    img_height: int = 0,
) -> None:
    """Draw a red arrow showing the robot's position and heading."""
    draw.ellipse(
        [px - 5, py - 5, px + 5, py + 5],
        fill=(220, 40, 40),
        outline=(180, 0, 0),
    )

    # The yaw is in standard math convention (CCW from +X in world frame).
    # In pixel coords, Y is flipped, so negate the Y component.
    tip_x = px + int(arrow_len * math.cos(yaw))
    tip_y = py - int(arrow_len * math.sin(yaw))
    draw.line([(px, py), (tip_x, tip_y)], fill=(220, 40, 40), width=3)

    # arrowhead
    head_angle = 0.5
    head_len = 7
    for sign in (-1, 1):
        hx = tip_x - int(head_len * math.cos(yaw + sign * head_angle))
        hy = tip_y + int(head_len * math.sin(yaw + sign * head_angle))
        draw.line([(tip_x, tip_y), (hx, hy)], fill=(220, 40, 40), width=2)


def _draw_destination(draw: ImageDraw.ImageDraw, px: int, py: int, radius: int = 12) -> None:
    """Draw a green star/cross marker at the destination."""
    draw.ellipse(
        [px - radius, py - radius, px + radius, py + radius],
        outline=(0, 200, 0),
        width=3,
    )
    draw.line([(px - radius, py), (px + radius, py)], fill=(0, 200, 0), width=2)
    draw.line([(px, py - radius), (px, py + radius)], fill=(0, 200, 0), width=2)


def _draw_source(draw: ImageDraw.ImageDraw, px: int, py: int, radius: int = 8) -> None:
    """Draw a blue circle at the source / start position."""
    draw.ellipse(
        [px - radius, py - radius, px + radius, py + radius],
        outline=(60, 60, 220),
        width=2,
    )
