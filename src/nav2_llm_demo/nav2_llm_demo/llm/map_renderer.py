"""Render an annotated top-down map image for the vision LLM.

Loads the PGM occupancy grid, draws the robot position/heading, the
destination marker, and the source marker, then returns a base64-encoded
PNG suitable for multimodal LLM messages.
"""

import base64
import io
import math
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

if TYPE_CHECKING:
    from .topology_graph import TopologyGraph


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


def render_graph_debug_map(
    map_yaml_path: str | Path,
    robot_x: float,
    robot_y: float,
    robot_yaw: float,
    graph: "TopologyGraph",
    *,
    route_nodes: list[str] | None = None,
    route_cursor: int = 0,
    active_edge_id: str = "",
    blocked_edge_ids: list[str] | None = None,
    source_x: float | None = None,
    source_y: float | None = None,
    dest_x: float | None = None,
    dest_y: float | None = None,
    output_size: int = 1200,
    caption_lines: list[str] | None = None,
) -> Image.Image:
    """Render a full debug map with graph overlays, labels, and route state."""
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
    font = ImageFont.load_default()

    blocked_edge_ids = blocked_edge_ids or []
    route_nodes = route_nodes or []

    route_edges = set()
    for from_node, to_node in zip(route_nodes, route_nodes[1:]):
        edge = graph.edge_between(from_node, to_node)
        if edge is not None:
            route_edges.add(edge.edge_id)

    for edge in graph.edges.values():
        from_node = graph.get_node(edge.from_node)
        to_node = graph.get_node(edge.to_node)
        if from_node is None or to_node is None:
            continue
        start_px = world_to_pixel(from_node.x, from_node.y, origin_x, origin_y, resolution, img_h)
        end_px = world_to_pixel(to_node.x, to_node.y, origin_x, origin_y, resolution, img_h)

        color = (110, 170, 230)
        width = 2
        if edge.edge_id in blocked_edge_ids or edge.status == "blocked":
            color = (220, 40, 40)
            width = 4
        elif edge.edge_id == active_edge_id:
            color = (255, 140, 0)
            width = 5
        elif edge.edge_id in route_edges:
            color = (255, 200, 0)
            width = 4

        draw.line([start_px, end_px], fill=color, width=width)

    for idx, node_id in enumerate(route_nodes):
        node = graph.get_node(node_id)
        if node is None:
            continue
        px, py = world_to_pixel(node.x, node.y, origin_x, origin_y, resolution, img_h)
        radius = 5
        fill = (255, 215, 0) if idx >= route_cursor else (140, 220, 140)
        if idx == route_cursor:
            fill = (255, 140, 0)
            radius = 7
        draw.ellipse([px - radius, py - radius, px + radius, py + radius], fill=fill, outline=(0, 0, 0))

    for node in graph.nodes.values():
        px, py = world_to_pixel(node.x, node.y, origin_x, origin_y, resolution, img_h)
        radius = 3
        outline = (0, 0, 0)
        fill = (70, 130, 255)
        if node.node_type == "start":
            fill = (60, 60, 220)
            radius = 6
        elif node.node_type == "goal":
            fill = (0, 200, 0)
            radius = 6
        elif node.node_type == "junction":
            fill = (150, 70, 220)
            radius = 5
        draw.ellipse([px - radius, py - radius, px + radius, py + radius], fill=fill, outline=outline)
        label = node.node_id
        text_x = px + 5
        text_y = py - 5
        text_width = max(1, int(draw.textlength(label, font=font)))
        text_height = 11
        draw.rectangle(
            [text_x - 1, text_y - 1, text_x + text_width + 2, text_y + text_height + 1],
            fill=(255, 255, 255),
            outline=(180, 180, 180),
        )
        draw.text((text_x, text_y), label, fill=(0, 0, 0), font=font)

    if dest_x is not None and dest_y is not None:
        dest_px, dest_py = world_to_pixel(dest_x, dest_y, origin_x, origin_y, resolution, img_h)
        _draw_destination(draw, dest_px, dest_py, radius=12)

    if source_x is not None and source_y is not None:
        src_px, src_py = world_to_pixel(source_x, source_y, origin_x, origin_y, resolution, img_h)
        _draw_source(draw, src_px, src_py, radius=8)

    robot_px, robot_py = world_to_pixel(robot_x, robot_y, origin_x, origin_y, resolution, img_h)
    _draw_robot(draw, robot_px, robot_py, robot_yaw, arrow_len=18, img_height=img_h)

    if caption_lines:
        margin = 8
        line_height = 14
        box_height = margin * 2 + line_height * len(caption_lines)
        draw.rectangle([0, 0, img_w, box_height], fill=(255, 255, 255), outline=(170, 170, 170))
        for idx, line in enumerate(caption_lines):
            draw.text((margin, margin + idx * line_height), line, fill=(0, 0, 0), font=font)

    return img.resize((output_size, output_size), Image.NEAREST)


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
