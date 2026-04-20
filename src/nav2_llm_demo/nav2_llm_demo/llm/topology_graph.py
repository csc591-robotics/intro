"""Topology graph primitives for LLM-assisted route planning.

This module gives the LLM a higher-level planning target than raw branch
choices. The graph is intentionally lightweight:

- nodes represent junctions, corridors, start, and goal
- edges represent traversable connections between nodes
- metadata can store map-derived or LLM-derived annotations

The graph is meant to be treated as a proposal that can be validated by
deterministic code before execution.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TopologyNode:
    """A topological waypoint or region in the route graph."""

    node_id: str
    node_type: str
    label: str = ""
    x: float | None = None
    y: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "node_type": self.node_type,
            "label": self.label,
            "x": self.x,
            "y": self.y,
            "metadata": dict(self.metadata),
        }


@dataclass
class TopologyEdge:
    """A directed or undirected connection between two topology nodes."""

    from_node: str
    to_node: str
    status: str = "unknown"
    edge_type: str = "corridor"
    cost: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "from_node": self.from_node,
            "to_node": self.to_node,
            "status": self.status,
            "edge_type": self.edge_type,
            "cost": self.cost,
            "metadata": dict(self.metadata),
        }


@dataclass
class TopologyGraph:
    """A compact graph representation for route-level planning."""

    nodes: dict[str, TopologyNode] = field(default_factory=dict)
    edges: list[TopologyEdge] = field(default_factory=list)
    start_node_id: str = ""
    goal_node_id: str = ""
    notes: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_node(
        self,
        node_id: str,
        node_type: str,
        *,
        label: str = "",
        x: float | None = None,
        y: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> TopologyNode:
        node = TopologyNode(
            node_id=node_id,
            node_type=node_type,
            label=label,
            x=x,
            y=y,
            metadata=dict(metadata or {}),
        )
        self.nodes[node_id] = node
        return node

    def add_edge(
        self,
        from_node: str,
        to_node: str,
        *,
        status: str = "unknown",
        edge_type: str = "corridor",
        cost: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> TopologyEdge:
        edge = TopologyEdge(
            from_node=from_node,
            to_node=to_node,
            status=status,
            edge_type=edge_type,
            cost=cost,
            metadata=dict(metadata or {}),
        )
        self.edges.append(edge)
        return edge

    def neighbors(self, node_id: str) -> list[TopologyEdge]:
        return [edge for edge in self.edges if edge.from_node == node_id]

    def edge_between(self, from_node: str, to_node: str) -> TopologyEdge | None:
        for edge in self.edges:
            if edge.from_node == from_node and edge.to_node == to_node:
                return edge
        return None

    def _iter_neighbors(self, node_id: str, *, allow_blocked: bool = False) -> list[tuple[str, TopologyEdge]]:
        neighbors: list[tuple[str, TopologyEdge]] = []
        for edge in self.neighbors(node_id):
            if not allow_blocked and edge.status == "blocked":
                continue
            neighbors.append((edge.to_node, edge))
            if edge.metadata.get("bidirectional") or edge.metadata.get("undirected"):
                reverse = TopologyEdge(
                    from_node=edge.to_node,
                    to_node=edge.from_node,
                    status=edge.status,
                    edge_type=edge.edge_type,
                    cost=edge.cost,
                    metadata=dict(edge.metadata),
                )
                neighbors.append((reverse.to_node, reverse))
        return neighbors

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

        queue: list[list[str]] = [[start]]
        visited = {start}

        while queue:
            path = queue.pop(0)
            node_id = path[-1]
            for next_node, edge in self._iter_neighbors(node_id, allow_blocked=allow_blocked):
                if next_node in visited or next_node not in self.nodes:
                    continue
                if not allow_blocked and edge.status == "blocked":
                    continue
                new_path = path + [next_node]
                if next_node == goal:
                    return new_path
                visited.add(next_node)
                queue.append(new_path)

        return []

    def _edge_action(self, edge: TopologyEdge | None) -> str:
        if edge is None:
            return "continue_route"
        metadata = edge.metadata or {}
        route_action = str(metadata.get("route_action", "")).strip()
        if route_action in {
            "continue_route",
            "take_left_branch",
            "take_right_branch",
            "backtrack",
        }:
            return route_action
        branch_side = str(
            metadata.get("branch_side")
            or metadata.get("side")
            or metadata.get("direction")
            or metadata.get("turn")
            or ""
        ).lower()
        if edge.edge_type == "backtrack" or route_action == "backtrack":
            return "backtrack"
        if edge.edge_type == "branch":
            if "left" in branch_side:
                return "take_left_branch"
            if "right" in branch_side:
                return "take_right_branch"
        return "continue_route"

    def _edge_repeat_count(self, edge: TopologyEdge | None) -> int:
        if edge is None:
            return 1
        metadata = edge.metadata or {}
        raw_steps = metadata.get("steps")
        if isinstance(raw_steps, int) and raw_steps > 0:
            return max(1, min(raw_steps, 5))
        if isinstance(raw_steps, float) and raw_steps > 0.0:
            return max(1, min(int(round(raw_steps)), 5))
        if isinstance(edge.cost, (int, float)) and edge.cost > 0.0:
            return max(1, min(int(round(edge.cost)), 5))
        return 1

    def path_to_route_sequence(self, path: list[str]) -> list[str]:
        if len(path) < 2:
            return []
        sequence: list[str] = []
        for from_node, to_node in zip(path, path[1:]):
            edge = self.edge_between(from_node, to_node)
            if edge is None:
                edge = self.edge_between(to_node, from_node)
            action = self._edge_action(edge)
            sequence.extend([action] * self._edge_repeat_count(edge))
        return sequence

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_node_id": self.start_node_id,
            "goal_node_id": self.goal_node_id,
            "notes": self.notes,
            "metadata": dict(self.metadata),
            "nodes": [node.to_dict() for node in self.nodes.values()],
            "edges": [edge.to_dict() for edge in self.edges],
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
            graph.nodes[node_id] = TopologyNode(
                node_id=node_id,
                node_type=str(node_payload.get("node_type", "junction")),
                label=str(node_payload.get("label", "")),
                x=node_payload.get("x"),
                y=node_payload.get("y"),
                metadata=dict(node_payload.get("metadata", {})),
            )

        for edge_payload in payload.get("edges", []):
            if not isinstance(edge_payload, dict):
                continue
            from_node = str(edge_payload.get("from_node", "")).strip()
            to_node = str(edge_payload.get("to_node", "")).strip()
            if not from_node or not to_node:
                continue
            graph.edges.append(
                TopologyEdge(
                    from_node=from_node,
                    to_node=to_node,
                    status=str(edge_payload.get("status", "unknown")),
                    edge_type=str(edge_payload.get("edge_type", "corridor")),
                    cost=edge_payload.get("cost"),
                    metadata=dict(edge_payload.get("metadata", {})),
                )
            )

        return graph
