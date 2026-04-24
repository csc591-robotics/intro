"""Deterministic topology extraction from occupancy maps."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import yaml

from .topology_graph import TopologyGraph


@dataclass
class OccupancyMap:
    """Convenience wrapper around a Nav2 occupancy-grid map."""

    map_yaml_path: str
    resolution: float
    origin_x: float
    origin_y: float
    grid: np.ndarray
    free_threshold: int = 200

    @classmethod
    def from_yaml(cls, map_yaml_path: str | Path) -> "OccupancyMap":
        map_yaml = Path(map_yaml_path)
        with map_yaml.open() as handle:
            meta = yaml.safe_load(handle)

        resolution = float(meta["resolution"])
        origin = meta.get("origin", [0.0, 0.0, 0.0])
        origin_x = float(origin[0])
        origin_y = float(origin[1])

        grid = np.array(Image.open(map_yaml.parent / meta["image"]).convert("L"))
        if int(meta.get("negate", 0)):
            grid = 255 - grid

        return cls(
            map_yaml_path=str(map_yaml),
            resolution=resolution,
            origin_x=origin_x,
            origin_y=origin_y,
            grid=grid,
        )

    @property
    def height(self) -> int:
        return int(self.grid.shape[0])

    @property
    def width(self) -> int:
        return int(self.grid.shape[1])

    def free_mask(self) -> np.ndarray:
        return self.grid >= self.free_threshold

    def world_to_grid(self, wx: float, wy: float) -> tuple[int, int]:
        col = int((wx - self.origin_x) / self.resolution)
        row = self.height - 1 - int((wy - self.origin_y) / self.resolution)
        return row, col

    def grid_to_world(self, row: int, col: int) -> tuple[float, float]:
        wx = self.origin_x + (float(col) + 0.5) * self.resolution
        wy = self.origin_y + (float(self.height - 1 - row) + 0.5) * self.resolution
        return wx, wy

    def grid_value(self, wx: float, wy: float) -> int | None:
        row, col = self.world_to_grid(wx, wy)
        if row < 0 or row >= self.height or col < 0 or col >= self.width:
            return None
        return int(self.grid[row, col])

    def footprint_is_clear(self, wx: float, wy: float, *, robot_radius_m: float) -> bool:
        sample_angles = [
            0.0,
            math.pi / 4.0,
            math.pi / 2.0,
            3.0 * math.pi / 4.0,
            math.pi,
            -3.0 * math.pi / 4.0,
            -math.pi / 2.0,
            -math.pi / 4.0,
        ]
        sample_points = [(wx, wy)]
        for angle in sample_angles:
            sample_points.append(
                (
                    wx + robot_radius_m * math.cos(angle),
                    wy + robot_radius_m * math.sin(angle),
                )
            )
        for sample_x, sample_y in sample_points:
            value = self.grid_value(sample_x, sample_y)
            if value is None or value < self.free_threshold:
                return False
        return True

    def is_segment_clear(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
        *,
        robot_radius_m: float,
    ) -> bool:
        step = max(self.resolution / 2.0, 0.03)
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        distance = math.hypot(dx, dy)
        if distance < 1e-6:
            return self.footprint_is_clear(start[0], start[1], robot_radius_m=robot_radius_m)

        samples = max(1, int(math.ceil(distance / step)))
        for idx in range(samples + 1):
            ratio = idx / samples
            wx = start[0] + ratio * dx
            wy = start[1] + ratio * dy
            if not self.footprint_is_clear(wx, wy, robot_radius_m=robot_radius_m):
                return False
        return True


@dataclass
class _AnchorCluster:
    pixel_id: int
    pixels: list[tuple[int, int]]
    node_type: str
    center: tuple[int, int]


@dataclass
class _CorridorTrace:
    start_anchor_id: int
    end_anchor_id: int
    pixels: list[tuple[int, int]]


class DeterministicTopologyBuilder:
    """Extract a topology graph from an occupancy map without using the LLM."""

    def __init__(
        self,
        occupancy_map: OccupancyMap,
        *,
        waypoint_spacing_m: float = 0.75,
        robot_radius_m: float = 0.11,
        special_node_connection_limit: int = 3,
        anchor_merge_distance_m: float = 0.75,
        preserve_junction_nodes: bool = True,
    ) -> None:
        self._occupancy_map = occupancy_map
        self._waypoint_spacing_m = max(0.25, waypoint_spacing_m)
        self._robot_radius_m = robot_radius_m
        self._special_node_connection_limit = max(1, special_node_connection_limit)
        self._anchor_merge_distance_m = max(0.0, anchor_merge_distance_m)
        self._preserve_junction_nodes = preserve_junction_nodes
        self._edge_counter = 0
        self._waypoint_counter = 0

    def build(
        self,
        *,
        source_pose: tuple[float, float] | None = None,
        goal_pose: tuple[float, float] | None = None,
    ) -> TopologyGraph:
        skeleton = self._skeletonize(self._occupancy_map.free_mask())
        anchor_clusters = self._extract_anchor_clusters(skeleton)
        graph = TopologyGraph()

        if not anchor_clusters:
            raise RuntimeError("Deterministic topology builder found no graph anchors in the map.")

        anchor_pixel_to_id: dict[tuple[int, int], int] = {}
        anchor_node_name: dict[int, str] = {}

        for cluster in anchor_clusters:
            world_x, world_y = self._occupancy_map.grid_to_world(*cluster.center)
            node_name = f"anchor_{cluster.pixel_id}"
            anchor_node_name[cluster.pixel_id] = node_name
            graph.add_node(
                node_id=node_name,
                node_type=cluster.node_type,
                x=world_x,
                y=world_y,
                label=node_name,
                metadata={"source": "deterministic_topology"},
            )
            for pixel in cluster.pixels:
                anchor_pixel_to_id[pixel] = cluster.pixel_id

        traces = self._trace_corridors(skeleton, anchor_pixel_to_id)
        if not traces:
            raise RuntimeError("Deterministic topology builder could not connect any graph anchors.")

        for trace in traces:
            start_node = anchor_node_name[trace.start_anchor_id]
            end_node = anchor_node_name[trace.end_anchor_id]
            self._emit_corridor(graph, start_node, end_node, trace.pixels)

        if source_pose is not None:
            self._attach_special_node(graph, "start", *source_pose)
        if goal_pose is not None:
            self._attach_special_node(graph, "goal", *goal_pose)

        pre_simplify = TopologyGraph.from_dict(graph.to_dict())
        self._simplify_graph(graph)
        self._merge_nearby_anchors(graph)
        self._simplify_graph(graph)

        reverted_collapse = False
        if self._is_degenerate_after_collapse(pre_simplify, graph):
            graph = pre_simplify
            reverted_collapse = True

        graph.metadata["build_stats"] = {
            "pre_nodes": len(pre_simplify.nodes),
            "pre_edges": len(pre_simplify.edges),
            "post_nodes": len(graph.nodes),
            "post_edges": len(graph.edges),
            "collapsed_reverted": reverted_collapse,
            "connectivity_ok": bool(
                graph.find_path(graph.start_node_id, graph.goal_node_id)
            ),
        }
        return graph

    def _is_degenerate_after_collapse(
        self,
        before: TopologyGraph,
        after: TopologyGraph,
    ) -> bool:
        before_non_special = self._non_special_node_count(before)
        after_non_special = self._non_special_node_count(after)
        before_path = bool(before.find_path(before.start_node_id, before.goal_node_id))
        after_path = bool(after.find_path(after.start_node_id, after.goal_node_id))

        if before_non_special > 0 and after_non_special == 0:
            return True
        if len(before.edges) > 1 and len(after.edges) <= 1:
            return True
        if before_path and not after_path:
            return True
        for special_node in ("start", "goal"):
            if special_node in after.nodes and self._non_self_neighbor_count(after, special_node) == 0:
                return True
        return False

    def _non_special_node_count(self, graph: TopologyGraph) -> int:
        return sum(1 for node in graph.nodes.values() if node.node_type not in {"start", "goal"})

    def _non_self_neighbor_count(self, graph: TopologyGraph, node_id: str) -> int:
        neighbors = set()
        for neighbor_id, _edge in graph.neighbors(node_id, allow_blocked=True):
            if neighbor_id == node_id:
                continue
            neighbors.add(neighbor_id)
        return len(neighbors)

    def _emit_corridor(
        self,
        graph: TopologyGraph,
        start_node_id: str,
        end_node_id: str,
        path_pixels: list[tuple[int, int]],
    ) -> None:
        if start_node_id == end_node_id:
            return

        start_node = graph.get_node(start_node_id)
        end_node = graph.get_node(end_node_id)
        if start_node is None or end_node is None:
            return

        polyline = [(start_node.x, start_node.y)]
        for row, col in path_pixels:
            polyline.append(self._occupancy_map.grid_to_world(row, col))
        polyline.append((end_node.x, end_node.y))
        sampled = self._sample_polyline(polyline, spacing_m=self._waypoint_spacing_m)
        self._add_segment_edge(
            graph,
            start_node_id,
            end_node_id,
            start_point=(start_node.x, start_node.y),
            end_point=(end_node.x, end_node.y),
            polyline=sampled,
        )

    def _add_segment_edge(
        self,
        graph: TopologyGraph,
        from_node_id: str,
        to_node_id: str,
        start_point: tuple[float, float],
        end_point: tuple[float, float],
        polyline: list[tuple[float, float]] | None = None,
    ) -> None:
        if polyline is None:
            polyline = [start_point, end_point]
        edge_id = f"edge_{self._edge_counter}"
        self._edge_counter += 1
        graph.add_edge(
            edge_id=edge_id,
            from_node=from_node_id,
            to_node=to_node_id,
            status="open",
            cost=self._polyline_length(polyline),
            metadata={
                "undirected": True,
                "polyline": polyline,
            },
        )

    def _attach_special_node(self, graph: TopologyGraph, node_id: str, wx: float, wy: float) -> None:
        graph.add_node(
            node_id=node_id,
            node_type=node_id,
            x=wx,
            y=wy,
            label=node_id,
            metadata={"source": "configured_pose"},
        )

        candidates: list[tuple[float, str]] = []
        for candidate in graph.nodes.values():
            if candidate.node_id == node_id:
                continue
            if not self._occupancy_map.is_segment_clear(
                (wx, wy),
                (candidate.x, candidate.y),
                robot_radius_m=self._robot_radius_m,
            ):
                continue
            distance = math.hypot(candidate.x - wx, candidate.y - wy)
            candidates.append((distance, candidate.node_id))

        if not candidates:
            # Fallback to the nearest *non-special* node. If we let nearest_node()
            # see the just-added start/goal node, it can select itself and create
            # a self-loop (start->start or goal->goal), disconnecting the graph.
            nearest = graph.nearest_node(
                wx,
                wy,
                include_types={"anchor", "junction", "corridor"},
            )
            if nearest is None:
                raise RuntimeError(f"Could not attach special node '{node_id}' to the graph.")
            candidates = [(math.hypot(nearest.x - wx, nearest.y - wy), nearest.node_id)]

        attached_neighbors: set[str] = set()
        for distance, neighbor_id in sorted(candidates)[: self._special_node_connection_limit]:
            if neighbor_id in attached_neighbors:
                continue
            attached_neighbors.add(neighbor_id)
            edge_id = f"edge_{self._edge_counter}"
            self._edge_counter += 1
            graph.add_edge(
                edge_id=edge_id,
                from_node=node_id,
                to_node=neighbor_id,
                status="open",
                cost=distance,
                metadata={
                    "undirected": True,
                    "polyline": [(wx, wy), (graph.nodes[neighbor_id].x, graph.nodes[neighbor_id].y)],
                },
            )

        if node_id == "start":
            graph.start_node_id = node_id
        elif node_id == "goal":
            graph.goal_node_id = node_id

    def _sample_polyline(
        self,
        polyline: list[tuple[float, float]],
        *,
        spacing_m: float,
    ) -> list[tuple[float, float]]:
        if len(polyline) <= 2:
            return polyline

        sampled = [polyline[0]]
        carry = 0.0
        for start, end in zip(polyline, polyline[1:]):
            dx = end[0] - start[0]
            dy = end[1] - start[1]
            segment_len = math.hypot(dx, dy)
            if segment_len < 1e-6:
                continue
            distance_along = spacing_m - carry
            while distance_along < segment_len:
                ratio = distance_along / segment_len
                sampled.append((start[0] + ratio * dx, start[1] + ratio * dy))
                distance_along += spacing_m
            carry = max(0.0, segment_len - (distance_along - spacing_m))
        sampled.append(polyline[-1])
        return sampled

    def _polyline_length(self, polyline: list[tuple[float, float]]) -> float:
        total = 0.0
        for start, end in zip(polyline, polyline[1:]):
            total += math.hypot(end[0] - start[0], end[1] - start[1])
        return total

    def _extract_anchor_clusters(self, skeleton: np.ndarray) -> list[_AnchorCluster]:
        height, width = skeleton.shape
        candidate_mask = np.zeros_like(skeleton, dtype=bool)

        for row in range(height):
            for col in range(width):
                if not skeleton[row, col]:
                    continue
                neighbor_count = len(self._skeleton_neighbors(skeleton, row, col))
                if neighbor_count == 1 or neighbor_count >= 3:
                    candidate_mask[row, col] = True

        visited = np.zeros_like(candidate_mask, dtype=bool)
        clusters: list[_AnchorCluster] = []
        cluster_id = 0

        for row in range(height):
            for col in range(width):
                if not candidate_mask[row, col] or visited[row, col]:
                    continue
                pixels: list[tuple[int, int]] = []
                queue = deque([(row, col)])
                visited[row, col] = True
                node_type = "dead_end"
                while queue:
                    cur_row, cur_col = queue.popleft()
                    pixels.append((cur_row, cur_col))
                    neighbor_count = len(self._skeleton_neighbors(skeleton, cur_row, cur_col))
                    if neighbor_count >= 3:
                        node_type = "junction"
                    for next_row in range(max(0, cur_row - 1), min(height, cur_row + 2)):
                        for next_col in range(max(0, cur_col - 1), min(width, cur_col + 2)):
                            if (next_row, next_col) == (cur_row, cur_col):
                                continue
                            if not candidate_mask[next_row, next_col] or visited[next_row, next_col]:
                                continue
                            visited[next_row, next_col] = True
                            queue.append((next_row, next_col))

                center = self._centroid_pixel(pixels)
                clusters.append(
                    _AnchorCluster(
                        pixel_id=cluster_id,
                        pixels=pixels,
                        node_type=node_type,
                        center=center,
                    )
                )
                cluster_id += 1

        return clusters

    def _simplify_graph(self, graph: TopologyGraph) -> None:
        """Collapse degree-2 non-special anchors into single corridor edges."""
        changed = True
        while changed:
            changed = False
            # Keep at least some intermediate graph structure for planning/debugging.
            if self._non_special_node_count(graph) <= 3:
                return
            for node_id in list(graph.nodes.keys()):
                node = graph.nodes.get(node_id)
                if node is None or node.node_type in {"start", "goal"}:
                    continue
                if self._preserve_junction_nodes and node.node_type == "junction":
                    continue
                neighbors = self._undirected_neighbors(graph, node_id)
                if len(neighbors) != 2:
                    continue

                (left_id, left_edge), (right_id, right_edge) = neighbors
                if left_id == right_id:
                    continue

                left_poly = self._oriented_edge_polyline(left_edge, left_id, node_id)
                right_poly = self._oriented_edge_polyline(right_edge, node_id, right_id)
                if not left_poly or not right_poly:
                    continue

                merged_poly = left_poly + right_poly[1:]
                merged_cost = self._polyline_length(merged_poly)
                existing_edge = graph.edge_between(left_id, right_id)

                del graph.edges[left_edge.edge_id]
                del graph.edges[right_edge.edge_id]
                del graph.nodes[node_id]

                if existing_edge is None:
                    edge_id = f"edge_{self._edge_counter}"
                    self._edge_counter += 1
                    graph.add_edge(
                        edge_id=edge_id,
                        from_node=left_id,
                        to_node=right_id,
                        status="open",
                        cost=merged_cost,
                        metadata={
                            "undirected": True,
                            "polyline": merged_poly,
                        },
                    )
                elif merged_cost < existing_edge.cost:
                    existing_edge.cost = merged_cost
                    existing_edge.metadata["polyline"] = merged_poly
                changed = True
                break

    def _merge_nearby_anchors(self, graph: TopologyGraph) -> None:
        """Merge spatially nearby anchor nodes into one representative."""
        if self._anchor_merge_distance_m <= 0.0:
            return

        while True:
            clusters = self._nearby_anchor_clusters(graph)
            clusters = [cluster for cluster in clusters if len(cluster) > 1]
            if not clusters:
                return
            for cluster in clusters:
                self._merge_anchor_cluster(graph, cluster)
            self._remove_duplicate_edges(graph)

    def _nearby_anchor_clusters(self, graph: TopologyGraph) -> list[list[str]]:
        eligible_ids = [
            node_id
            for node_id, node in graph.nodes.items()
            if node.node_type not in {"start", "goal"}
        ]
        unvisited = set(eligible_ids)
        clusters: list[list[str]] = []

        while unvisited:
            seed = unvisited.pop()
            cluster = [seed]
            queue = [seed]
            while queue:
                current_id = queue.pop(0)
                current_node = graph.nodes[current_id]
                for other_id in list(unvisited):
                    other_node = graph.nodes[other_id]
                    distance = math.hypot(current_node.x - other_node.x, current_node.y - other_node.y)
                    if distance > self._anchor_merge_distance_m:
                        continue
                    queue.append(other_id)
                    cluster.append(other_id)
                    unvisited.remove(other_id)
            clusters.append(cluster)

        return clusters

    def _merge_anchor_cluster(self, graph: TopologyGraph, cluster_node_ids: list[str]) -> None:
        if len(cluster_node_ids) <= 1:
            return

        centroid_x = sum(graph.nodes[node_id].x for node_id in cluster_node_ids) / len(cluster_node_ids)
        centroid_y = sum(graph.nodes[node_id].y for node_id in cluster_node_ids) / len(cluster_node_ids)
        representative_id = min(
            cluster_node_ids,
            key=lambda node_id: math.hypot(graph.nodes[node_id].x - centroid_x, graph.nodes[node_id].y - centroid_y),
        )
        representative_node = graph.nodes[representative_id]
        representative_node.node_type = (
            "junction"
            if any(graph.nodes[node_id].node_type == "junction" for node_id in cluster_node_ids)
            else representative_node.node_type
        )

        for node_id in cluster_node_ids:
            if node_id == representative_id or node_id not in graph.nodes:
                continue
            merged_node = graph.nodes[node_id]
            for edge in list(graph.edges.values()):
                if edge.from_node != node_id and edge.to_node != node_id:
                    continue

                if edge.from_node == node_id:
                    edge.from_node = representative_id
                    polyline = list(edge.metadata.get("polyline", []))
                    if polyline:
                        polyline[0] = (representative_node.x, representative_node.y)
                        edge.metadata["polyline"] = polyline
                if edge.to_node == node_id:
                    edge.to_node = representative_id
                    polyline = list(edge.metadata.get("polyline", []))
                    if polyline:
                        polyline[-1] = (representative_node.x, representative_node.y)
                        edge.metadata["polyline"] = polyline

                if edge.from_node == edge.to_node:
                    del graph.edges[edge.edge_id]
                    continue

            representative_node.metadata.setdefault("merged_anchor_ids", []).append(node_id)
            representative_node.metadata.setdefault("merged_anchor_positions", []).append((merged_node.x, merged_node.y))
            del graph.nodes[node_id]

    def _remove_duplicate_edges(self, graph: TopologyGraph) -> None:
        best_edges: dict[tuple[str, str], str] = {}
        for edge_id, edge in list(graph.edges.items()):
            key = tuple(sorted((edge.from_node, edge.to_node)))
            current_best_id = best_edges.get(key)
            if current_best_id is None:
                best_edges[key] = edge_id
                continue
            current_best = graph.edges[current_best_id]
            if edge.cost < current_best.cost:
                del graph.edges[current_best_id]
                best_edges[key] = edge_id
            else:
                del graph.edges[edge_id]

    def _undirected_neighbors(
        self,
        graph: TopologyGraph,
        node_id: str,
    ) -> list[tuple[str, Any]]:
        neighbors: list[tuple[str, Any]] = []
        for edge in graph.edges.values():
            if edge.from_node == node_id:
                neighbors.append((edge.to_node, edge))
            elif edge.is_undirected() and edge.to_node == node_id:
                neighbors.append((edge.from_node, edge))
        return neighbors

    def _oriented_edge_polyline(
        self,
        edge: Any,
        start_node_id: str,
        end_node_id: str,
    ) -> list[tuple[float, float]]:
        raw_polyline = list(edge.metadata.get("polyline", []))
        if not raw_polyline:
            start_node = edge.from_node
            end_node = edge.to_node
            if start_node == start_node_id and end_node == end_node_id:
                return []
        polyline = [(float(point[0]), float(point[1])) for point in raw_polyline]
        if edge.from_node == start_node_id and edge.to_node == end_node_id:
            return polyline
        if edge.is_undirected() and edge.from_node == end_node_id and edge.to_node == start_node_id:
            return list(reversed(polyline))
        return polyline

    def _trace_corridors(
        self,
        skeleton: np.ndarray,
        anchor_pixel_to_id: dict[tuple[int, int], int],
    ) -> list[_CorridorTrace]:
        visited_segments: set[tuple[tuple[int, int], tuple[int, int]]] = set()
        traces: list[_CorridorTrace] = []

        for anchor_pixel, anchor_id in anchor_pixel_to_id.items():
            row, col = anchor_pixel
            for neighbor in self._skeleton_neighbors(skeleton, row, col):
                if neighbor in anchor_pixel_to_id and anchor_pixel_to_id[neighbor] == anchor_id:
                    continue
                segment_key = self._segment_key(anchor_pixel, neighbor)
                if segment_key in visited_segments:
                    continue

                prev = anchor_pixel
                current = neighbor
                corridor_pixels: list[tuple[int, int]] = []
                end_anchor_id: int | None = None

                while True:
                    visited_segments.add(self._segment_key(prev, current))
                    corridor_pixels.append(current)
                    if current in anchor_pixel_to_id and anchor_pixel_to_id[current] != anchor_id:
                        end_anchor_id = anchor_pixel_to_id[current]
                        break

                    next_neighbors = [
                        candidate
                        for candidate in self._skeleton_neighbors(skeleton, current[0], current[1])
                        if candidate != prev
                    ]
                    if not next_neighbors:
                        break
                    if len(next_neighbors) > 1:
                        break
                    prev, current = current, next_neighbors[0]

                if end_anchor_id is None or end_anchor_id == anchor_id:
                    continue
                traces.append(
                    _CorridorTrace(
                        start_anchor_id=anchor_id,
                        end_anchor_id=end_anchor_id,
                        pixels=corridor_pixels[:-1],
                    )
                )

        deduped: dict[frozenset[int], _CorridorTrace] = {}
        for trace in traces:
            key = frozenset((trace.start_anchor_id, trace.end_anchor_id))
            if key not in deduped or len(trace.pixels) < len(deduped[key].pixels):
                deduped[key] = trace
        return list(deduped.values())

    def _segment_key(
        self,
        first: tuple[int, int],
        second: tuple[int, int],
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        return tuple(sorted((first, second)))

    def _centroid_pixel(self, pixels: list[tuple[int, int]]) -> tuple[int, int]:
        row = int(round(sum(pixel[0] for pixel in pixels) / len(pixels)))
        col = int(round(sum(pixel[1] for pixel in pixels) / len(pixels)))
        return row, col

    def _skeleton_neighbors(self, skeleton: np.ndarray, row: int, col: int) -> list[tuple[int, int]]:
        height, width = skeleton.shape
        result: list[tuple[int, int]] = []
        for next_row in range(max(0, row - 1), min(height, row + 2)):
            for next_col in range(max(0, col - 1), min(width, col + 2)):
                if (next_row, next_col) == (row, col):
                    continue
                if skeleton[next_row, next_col]:
                    result.append((next_row, next_col))
        return result

    def _skeletonize(self, free_mask: np.ndarray) -> np.ndarray:
        skeleton = free_mask.astype(np.uint8).copy()
        changed = True
        while changed:
            changed = False
            for step in (0, 1):
                removable: list[tuple[int, int]] = []
                for row in range(1, skeleton.shape[0] - 1):
                    for col in range(1, skeleton.shape[1] - 1):
                        if skeleton[row, col] != 1:
                            continue
                        neighbors = self._zhang_suen_neighbors(skeleton, row, col)
                        neighbor_sum = sum(neighbors)
                        if neighbor_sum < 2 or neighbor_sum > 6:
                            continue
                        if self._transition_count(neighbors) != 1:
                            continue
                        p2, p3, p4, p5, p6, p7, p8, p9 = neighbors
                        if step == 0:
                            if p2 * p4 * p6 != 0:
                                continue
                            if p4 * p6 * p8 != 0:
                                continue
                        else:
                            if p2 * p4 * p8 != 0:
                                continue
                            if p2 * p6 * p8 != 0:
                                continue
                        removable.append((row, col))
                if removable:
                    changed = True
                    for row, col in removable:
                        skeleton[row, col] = 0
        return skeleton.astype(bool)

    def _zhang_suen_neighbors(self, skeleton: np.ndarray, row: int, col: int) -> list[int]:
        return [
            int(skeleton[row - 1, col]),
            int(skeleton[row - 1, col + 1]),
            int(skeleton[row, col + 1]),
            int(skeleton[row + 1, col + 1]),
            int(skeleton[row + 1, col]),
            int(skeleton[row + 1, col - 1]),
            int(skeleton[row, col - 1]),
            int(skeleton[row - 1, col - 1]),
        ]

    def _transition_count(self, neighbors: list[int]) -> int:
        cyclic = neighbors + [neighbors[0]]
        return sum(1 for current, nxt in zip(cyclic, cyclic[1:]) if current == 0 and nxt == 1)
