"""Topology graph primitives for deterministic map extraction and LLM routing."""

from __future__ import annotations

import heapq
import json
import math
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TopologyNode:
    """A topological waypoint in the navigation graph."""

    node_id: str
    node_type: str
    x: float
    y: float
    label: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "node_type": self.node_type,
            "x": self.x,
            "y": self.y,
            "label": self.label,
            "metadata": dict(self.metadata),
        }


@dataclass
class TopologyEdge:
    """A traversable connection between two graph nodes."""

    edge_id: str
    from_node: str
    to_node: str
    status: str = "open"
    cost: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def is_undirected(self) -> bool:
        return bool(self.metadata.get("undirected", True))

    def to_dict(self) -> dict[str, Any]:
        return {
            "edge_id": self.edge_id,
            "from_node": self.from_node,
            "to_node": self.to_node,
            "status": self.status,
            "cost": self.cost,
            "metadata": dict(self.metadata),
        }


@dataclass
class TopologyGraph:
    """A weighted graph used for routing and blocked-edge replanning."""

    nodes: dict[str, TopologyNode] = field(default_factory=dict)
    edges: dict[str, TopologyEdge] = field(default_factory=dict)
    start_node_id: str = ""
    goal_node_id: str = ""
    notes: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_node(
        self,
        node_id: str,
        node_type: str,
        *,
        x: float,
        y: float,
        label: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> TopologyNode:
        node = TopologyNode(
            node_id=node_id,
            node_type=node_type,
            x=float(x),
            y=float(y),
            label=label,
            metadata=dict(metadata or {}),
        )
        self.nodes[node_id] = node
        return node

    def add_edge(
        self,
        edge_id: str,
        from_node: str,
        to_node: str,
        *,
        status: str = "open",
        cost: float = 1.0,
        metadata: dict[str, Any] | None = None,
    ) -> TopologyEdge:
        edge = TopologyEdge(
            edge_id=edge_id,
            from_node=from_node,
            to_node=to_node,
            status=status,
            cost=float(cost),
            metadata=dict(metadata or {}),
        )
        self.edges[edge_id] = edge
        return edge

    def get_node(self, node_id: str) -> TopologyNode | None:
        return self.nodes.get(node_id)

    def get_edge(self, edge_id: str) -> TopologyEdge | None:
        return self.edges.get(edge_id)

    def edge_between(self, from_node: str, to_node: str) -> TopologyEdge | None:
        for edge in self.edges.values():
            if edge.from_node == from_node and edge.to_node == to_node:
                return edge
            if edge.is_undirected() and edge.from_node == to_node and edge.to_node == from_node:
                return edge
        return None

    def set_edge_status(self, edge_id: str, status: str) -> None:
        edge = self.edges.get(edge_id)
        if edge is not None:
            edge.status = status

    def set_edge_status_between(self, from_node: str, to_node: str, status: str) -> bool:
        edge = self.edge_between(from_node, to_node)
        if edge is None:
            return False
        edge.status = status
        return True

    def blocked_edge_ids(self) -> list[str]:
        return sorted(edge.edge_id for edge in self.edges.values() if edge.status == "blocked")

    def neighbors(self, node_id: str, *, allow_blocked: bool = False) -> list[tuple[str, TopologyEdge]]:
        result: list[tuple[str, TopologyEdge]] = []
        for edge in self.edges.values():
            if not allow_blocked and edge.status == "blocked":
                continue
            if edge.from_node == node_id:
                result.append((edge.to_node, edge))
            elif edge.is_undirected() and edge.to_node == node_id:
                result.append((edge.from_node, edge))
        return result

    def nearest_node(self, x: float, y: float, *, include_types: set[str] | None = None) -> TopologyNode | None:
        best_node: TopologyNode | None = None
        best_dist = math.inf
        for node in self.nodes.values():
            if include_types is not None and node.node_type not in include_types:
                continue
            dist = math.hypot(node.x - x, node.y - y)
            if dist < best_dist:
                best_dist = dist
                best_node = node
        return best_node

    def find_path(
        self,
        start_node_id: str | None = None,
        goal_node_id: str | None = None,
        *,
        allow_blocked: bool = False,
    ) -> list[str]:
        start = (start_node_id or self.start_node_id).strip()
        goal = (goal_node_id or self.goal_node_id).strip()
        if not start or not goal or start not in self.nodes or goal not in self.nodes:
            return []
        if start == goal:
            return [start]

        frontier: list[tuple[float, str]] = [(0.0, start)]
        came_from: dict[str, str | None] = {start: None}
        costs: dict[str, float] = {start: 0.0}

        while frontier:
            current_cost, node_id = heapq.heappop(frontier)
            if node_id == goal:
                break
            if current_cost > costs.get(node_id, math.inf):
                continue
            for neighbor_id, edge in self.neighbors(node_id, allow_blocked=allow_blocked):
                next_cost = current_cost + max(0.001, float(edge.cost))
                if next_cost >= costs.get(neighbor_id, math.inf):
                    continue
                costs[neighbor_id] = next_cost
                came_from[neighbor_id] = node_id
                heapq.heappush(frontier, (next_cost, neighbor_id))

        if goal not in came_from:
            return []

        path: list[str] = []
        cursor: str | None = goal
        while cursor is not None:
            path.append(cursor)
            cursor = came_from[cursor]
        path.reverse()
        return path

    def validate_path(
        self,
        path_nodes: list[str],
        *,
        start_node_id: str | None = None,
        goal_node_id: str | None = None,
        allow_blocked: bool = False,
    ) -> tuple[bool, str]:
        if not path_nodes:
            return False, "Path is empty."
        for node_id in path_nodes:
            if node_id not in self.nodes:
                return False, f"Unknown node '{node_id}'."

        expected_start = (start_node_id or self.start_node_id).strip()
        expected_goal = (goal_node_id or self.goal_node_id).strip()

        if expected_start and path_nodes[0] != expected_start:
            return False, f"Path must start at '{expected_start}', got '{path_nodes[0]}'."
        if expected_goal and path_nodes[-1] != expected_goal:
            return False, f"Path must end at '{expected_goal}', got '{path_nodes[-1]}'."

        for from_node, to_node in zip(path_nodes, path_nodes[1:]):
            edge = self.edge_between(from_node, to_node)
            if edge is None:
                return False, f"No edge exists between '{from_node}' and '{to_node}'."
            if not allow_blocked and edge.status == "blocked":
                return False, f"Edge '{edge.edge_id}' between '{from_node}' and '{to_node}' is blocked."
        return True, "ok"

    def path_cost(self, path_nodes: list[str]) -> float:
        cost = 0.0
        for from_node, to_node in zip(path_nodes, path_nodes[1:]):
            edge = self.edge_between(from_node, to_node)
            if edge is None:
                return math.inf
            cost += edge.cost
        return cost

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_node_id": self.start_node_id,
            "goal_node_id": self.goal_node_id,
            "notes": self.notes,
            "metadata": dict(self.metadata),
            "nodes": [node.to_dict() for node in self.nodes.values()],
            "edges": [edge.to_dict() for edge in self.edges.values()],
        }

    def to_compact_dict(
        self,
        *,
        include_coordinates: bool = True,
        round_digits: int = 2,
    ) -> dict[str, Any]:
        nodes: list[dict[str, Any]] = []
        for node in self.nodes.values():
            payload = {
                "node_id": node.node_id,
                "node_type": node.node_type,
            }
            if include_coordinates:
                payload["x"] = round(float(node.x), round_digits)
                payload["y"] = round(float(node.y), round_digits)
            nodes.append(payload)

        edges: list[dict[str, Any]] = []
        for edge in self.edges.values():
            edges.append(
                {
                    "edge_id": edge.edge_id,
                    "from_node": edge.from_node,
                    "to_node": edge.to_node,
                    "status": edge.status,
                    "cost": round(float(edge.cost), round_digits),
                    "undirected": edge.is_undirected(),
                }
            )

        return {
            "start_node_id": self.start_node_id,
            "goal_node_id": self.goal_node_id,
            "notes": self.notes,
            "nodes": nodes,
            "edges": edges,
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TopologyGraph":
        graph = cls(
            start_node_id=str(payload.get("start_node_id", "")),
            goal_node_id=str(payload.get("goal_node_id", "")),
            notes=str(payload.get("notes", "")),
            metadata=dict(payload.get("metadata", {})),
        )

        for node_payload in payload.get("nodes", []):
            if not isinstance(node_payload, dict):
                continue
            node_id = str(node_payload.get("node_id", "")).strip()
            if not node_id:
                continue
            graph.add_node(
                node_id=node_id,
                node_type=str(node_payload.get("node_type", "corridor")),
                x=float(node_payload.get("x", 0.0)),
                y=float(node_payload.get("y", 0.0)),
                label=str(node_payload.get("label", "")),
                metadata=dict(node_payload.get("metadata", {})),
            )

        for edge_payload in payload.get("edges", []):
            if not isinstance(edge_payload, dict):
                continue
            edge_id = str(edge_payload.get("edge_id", "")).strip()
            from_node = str(edge_payload.get("from_node", "")).strip()
            to_node = str(edge_payload.get("to_node", "")).strip()
            if not edge_id or not from_node or not to_node:
                continue
            graph.add_edge(
                edge_id=edge_id,
                from_node=from_node,
                to_node=to_node,
                status=str(edge_payload.get("status", "open")),
                cost=float(edge_payload.get("cost", 1.0)),
                metadata=dict(edge_payload.get("metadata", {})),
            )

        return graph
