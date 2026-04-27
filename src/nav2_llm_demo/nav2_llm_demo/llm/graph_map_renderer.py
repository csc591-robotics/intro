"""Graph-overlay debug renderer for the topology-graph route planner.

Lives in its own module (rather than inside ``map_renderer.py``) so the
debug overlays for flow_7 don't collide with the LLM-facing renderers
maintained for flows 1-5. Re-uses ``map_renderer``'s primitives
(``load_map_yaml``, ``load_pgm``, ``world_to_pixel`` and the
robot/destination/source markers) so the visual style stays consistent
across flows.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .map_renderer import (
    _draw_destination,
    _draw_robot,
    _draw_source,
    load_map_yaml,
    load_pgm,
    world_to_pixel,
)

if TYPE_CHECKING:
    from .topology_graph import TopologyGraph


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


__all__ = ["render_graph_debug_map"]
