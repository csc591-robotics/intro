"""Deterministic A* path planner for flow_5.

Pure-Python (numpy + Pillow + heapq), no ROS or LangChain dependency,
so it can be unit-tested standalone on the host.

Pipeline:
    1. Load the PGM + YAML metadata via ``map_renderer``.
    2. Threshold to a binary obstacle grid.
    3. Inflate obstacles by ``inflation_px`` so the planned path keeps
       the robot's body at least ~``inflation_px * resolution`` meters
       away from any wall (roughly the TurtleBot3 Burger radius).
    4. Convert source/destination world coords to pixels and snap each
       to the nearest non-obstacle pixel (handles spawning slightly
       inside an inflated wall).
    5. Run 8-connected A* with euclidean heuristic. Diagonals cost
       sqrt(2); cardinals cost 1.0.
    6. Reconstruct the pixel path. Down-sample to roughly evenly spaced
       waypoints (every ``waypoint_spacing_m`` meters of running path
       length). Always include both endpoints.
    7. Convert waypoints back to map-frame meters and return.

If A* fails (no path), returns an empty list and the caller decides
what to do.
"""

from __future__ import annotations

import heapq
import math
from collections import deque
from pathlib import Path

import numpy as np

from ..map_renderer import load_map_yaml, load_pgm


# 8-connected neighbours: (dx, dy, cost)
_NEIGHBOURS: list[tuple[int, int, float]] = [
    (-1,  0, 1.0), (1,  0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
    (-1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)),
    (1, -1, math.sqrt(2)), (1, 1, math.sqrt(2)),
]


def _world_to_pixel(
    wx: float, wy: float,
    origin_x: float, origin_y: float,
    resolution: float, img_height: int,
) -> tuple[int, int]:
    px = int((wx - origin_x) / resolution)
    py = img_height - 1 - int((wy - origin_y) / resolution)
    return px, py


def _pixel_to_world(
    px: int, py: int,
    origin_x: float, origin_y: float,
    resolution: float, img_height: int,
) -> tuple[float, float]:
    wx = origin_x + (px + 0.5) * resolution
    wy = origin_y + (img_height - 1 - py + 0.5) * resolution
    return wx, wy


def _build_obstacle_grid(grid: np.ndarray, occ_threshold: int = 100) -> np.ndarray:
    """Return a bool array; True = obstacle, False = free."""
    return grid <= occ_threshold


def _inflate(obstacle: np.ndarray, radius_px: int) -> np.ndarray:
    """Binary dilation by ``radius_px`` cells using a chebyshev kernel.

    No scipy dependency: we shift the array in each direction up to
    ``radius_px`` and OR together. O(radius_px^2) shifts, fine for a few
    hundred-pixel maps and small radii.
    """
    if radius_px <= 0:
        return obstacle.copy()
    dilated = obstacle.copy()
    h, w = obstacle.shape
    for dy in range(-radius_px, radius_px + 1):
        for dx in range(-radius_px, radius_px + 1):
            if dy == 0 and dx == 0:
                continue
            sy0 = max(0, dy)
            sy1 = min(h, h + dy)
            sx0 = max(0, dx)
            sx1 = min(w, w + dx)
            dy0 = max(0, -dy)
            dy1 = dy0 + (sy1 - sy0)
            dx0 = max(0, -dx)
            dx1 = dx0 + (sx1 - sx0)
            dilated[sy0:sy1, sx0:sx1] |= obstacle[dy0:dy1, dx0:dx1]
    return dilated


def _snap_to_free(
    obstacle: np.ndarray, px: int, py: int, max_radius: int = 50,
) -> tuple[int, int] | None:
    """If ``(px, py)`` is inside an obstacle, BFS outward to the nearest
    free pixel (returning its coords). Returns ``None`` if no free
    pixel exists within ``max_radius``."""
    h, w = obstacle.shape
    if not (0 <= px < w and 0 <= py < h):
        return None
    if not obstacle[py, px]:
        return px, py
    visited = np.zeros_like(obstacle, dtype=bool)
    q: deque[tuple[int, int, int]] = deque()
    q.append((px, py, 0))
    visited[py, px] = True
    while q:
        x, y, d = q.popleft()
        if d > max_radius:
            return None
        for dx, dy, _ in _NEIGHBOURS:
            nx, ny = x + dx, y + dy
            if not (0 <= nx < w and 0 <= ny < h):
                continue
            if visited[ny, nx]:
                continue
            visited[ny, nx] = True
            if not obstacle[ny, nx]:
                return nx, ny
            q.append((nx, ny, d + 1))
    return None


def _astar(
    obstacle: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
) -> list[tuple[int, int]]:
    """Standard 8-connected A* on a binary obstacle grid.

    Coordinates are (px, py). Returns the pixel path including both
    endpoints, or [] if no path is found.
    """
    h, w = obstacle.shape
    sx, sy = start
    gx, gy = goal
    if obstacle[sy, sx] or obstacle[gy, gx]:
        return []

    def heuristic(x: int, y: int) -> float:
        return math.hypot(gx - x, gy - y)

    open_heap: list[tuple[float, int, int, int]] = []
    counter = 0
    g_score: dict[tuple[int, int], float] = {start: 0.0}
    came_from: dict[tuple[int, int], tuple[int, int]] = {}
    heapq.heappush(open_heap, (heuristic(sx, sy), counter, sx, sy))

    while open_heap:
        _, _, x, y = heapq.heappop(open_heap)
        if (x, y) == (gx, gy):
            path: list[tuple[int, int]] = [(x, y)]
            while path[-1] in came_from:
                path.append(came_from[path[-1]])
            path.reverse()
            return path
        gxy = g_score[(x, y)]
        for dx, dy, cost in _NEIGHBOURS:
            nx, ny = x + dx, y + dy
            if not (0 <= nx < w and 0 <= ny < h):
                continue
            if obstacle[ny, nx]:
                continue
            tentative = gxy + cost
            if tentative < g_score.get((nx, ny), math.inf):
                g_score[(nx, ny)] = tentative
                came_from[(nx, ny)] = (x, y)
                counter += 1
                heapq.heappush(
                    open_heap,
                    (tentative + heuristic(nx, ny), counter, nx, ny),
                )
    return []


def _downsample_pixel_path(
    pixel_path: list[tuple[int, int]],
    resolution: float,
    spacing_m: float,
) -> list[tuple[int, int]]:
    """Keep pixels along the path so consecutive waypoints are at least
    ``spacing_m`` meters apart. Always include first and last."""
    if not pixel_path:
        return []
    spacing_px = max(1.0, spacing_m / max(resolution, 1e-6))
    out = [pixel_path[0]]
    accum = 0.0
    prev = pixel_path[0]
    for cur in pixel_path[1:-1]:
        step = math.hypot(cur[0] - prev[0], cur[1] - prev[1])
        accum += step
        prev = cur
        if accum >= spacing_px:
            out.append(cur)
            accum = 0.0
    if pixel_path[-1] != out[-1]:
        out.append(pixel_path[-1])
    return out


def plan_astar_path(
    map_yaml_path: str | Path,
    src_xy: tuple[float, float],
    dst_xy: tuple[float, float],
    inflation_px: int = 3,
    waypoint_spacing_m: float = 0.6,
    occ_threshold: int = 100,
) -> list[tuple[float, float]]:
    """Plan a path from ``src_xy`` to ``dst_xy`` (map-frame meters).

    Returns a list of ``(x, y)`` waypoints in the same map frame. Empty
    list if no path exists or the inputs are out-of-bounds.
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
    if negate:
        grid = 255 - grid
    img_h, img_w = grid.shape

    obstacle = _build_obstacle_grid(grid, occ_threshold=occ_threshold)
    inflated = _inflate(obstacle, inflation_px)

    sx_px, sy_px = _world_to_pixel(
        src_xy[0], src_xy[1], origin_x, origin_y, resolution, img_h,
    )
    dx_px, dy_px = _world_to_pixel(
        dst_xy[0], dst_xy[1], origin_x, origin_y, resolution, img_h,
    )

    snapped_start = _snap_to_free(inflated, sx_px, sy_px)
    snapped_goal = _snap_to_free(inflated, dx_px, dy_px)
    if snapped_start is None or snapped_goal is None:
        return []

    pixel_path = _astar(inflated, snapped_start, snapped_goal)
    if not pixel_path:
        return []

    sampled_pixels = _downsample_pixel_path(
        pixel_path, resolution, waypoint_spacing_m,
    )
    waypoints: list[tuple[float, float]] = [
        _pixel_to_world(px, py, origin_x, origin_y, resolution, img_h)
        for (px, py) in sampled_pixels
    ]
    return waypoints


__all__ = ["plan_astar_path"]
