"""ROS 2 node for graph-based LLM navigation over a deterministic topology map."""

from __future__ import annotations

import json
import math
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from geometry_msgs.msg import TransformStamped, Twist
from nav_msgs.msg import Odometry
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
import tf2_ros

from .llm.llm_agent import build_agent
from .llm.map_renderer import render_graph_debug_map
from .llm.topology_builder import DeterministicTopologyBuilder, OccupancyMap
from .llm.topology_graph import TopologyGraph


def _yaw_from_quat(qx: float, qy: float, qz: float, qw: float) -> float:
    siny = 2.0 * (qw * qz + qx * qy)
    cosy = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny, cosy)


def _quat_from_yaw(yaw: float) -> tuple[float, float, float, float]:
    return 0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)


def _compose_pose(
    first: tuple[float, float, float],
    second: tuple[float, float, float],
) -> tuple[float, float, float]:
    fx, fy, fyaw = first
    sx, sy, syaw = second
    cos_yaw = math.cos(fyaw)
    sin_yaw = math.sin(fyaw)
    return (
        fx + cos_yaw * sx - sin_yaw * sy,
        fy + sin_yaw * sx + cos_yaw * sy,
        math.atan2(math.sin(fyaw + syaw), math.cos(fyaw + syaw)),
    )


def _invert_pose(pose: tuple[float, float, float]) -> tuple[float, float, float]:
    x, y, yaw = pose
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    return (
        -cos_yaw * x - sin_yaw * y,
        sin_yaw * x - cos_yaw * y,
        -yaw,
    )


@dataclass
class NavigationStepResult:
    status: str
    message: str


class LlmAgentNode(Node):
    """Graph-path navigation node with deterministic occupancy-map execution."""

    def __init__(self) -> None:
        super().__init__("llm_agent_node")

        self.declare_parameter("source_x", 0.0)
        self.declare_parameter("source_y", 0.0)
        self.declare_parameter("source_yaw", 0.0)
        self.declare_parameter("dest_x", 0.0)
        self.declare_parameter("dest_y", 0.0)
        self.declare_parameter("dest_yaw", 0.0)
        self.declare_parameter("map_yaml_path", "")
        self.declare_parameter("goal_tolerance_m", 0.5)
        self.declare_parameter("node_reach_tolerance_m", 0.30)
        self.declare_parameter("graph_waypoint_spacing_m", 0.75)
        self.declare_parameter("anchor_merge_distance_m", 0.75)
        self.declare_parameter("max_agent_steps", 240)
        self.declare_parameter("max_replans", 5)
        self.declare_parameter("edge_stall_step_limit", 6)
        self.declare_parameter("edge_progress_epsilon_m", 0.05)
        self.declare_parameter("linear_speed", 0.12)
        self.declare_parameter("angular_speed", 0.45)
        self.declare_parameter("max_rotation_step_deg", 20.0)
        self.declare_parameter("max_forward_step_m", 0.25)
        self.declare_parameter("heading_alignment_tolerance_deg", 10.0)
        self.declare_parameter("local_recovery_step_m", 0.18)
        self.declare_parameter("local_recovery_heading_offsets_deg", "15,30,45")
        self.declare_parameter("local_recovery_retry_limit", 3)
        self.declare_parameter("status_topic", "/navigation_status")
        self.declare_parameter("move_timeout_sec", 20.0)
        self.declare_parameter("base_frame", "base_footprint")
        self.declare_parameter("robot_radius_m", 0.11)

        self._source_x = self._float("source_x")
        self._source_y = self._float("source_y")
        self._source_yaw = self._float("source_yaw")
        self._dest_x = self._float("dest_x")
        self._dest_y = self._float("dest_y")
        self._dest_yaw = self._float("dest_yaw")
        self._map_yaml = self._str("map_yaml_path")
        self._goal_tolerance_m = self._float("goal_tolerance_m")
        self._node_reach_tolerance_m = self._float("node_reach_tolerance_m")
        self._graph_waypoint_spacing_m = self._float("graph_waypoint_spacing_m")
        self._anchor_merge_distance_m = self._float("anchor_merge_distance_m")
        self._max_agent_steps = self._int("max_agent_steps")
        self._max_replans = self._int("max_replans")
        self._edge_stall_step_limit = self._int("edge_stall_step_limit")
        self._edge_progress_epsilon_m = self._float("edge_progress_epsilon_m")
        self._linear_speed = self._float("linear_speed")
        self._angular_speed = self._float("angular_speed")
        self._max_rotation_step_deg = self._float("max_rotation_step_deg")
        self._max_forward_step_m = self._float("max_forward_step_m")
        self._heading_alignment_tolerance_deg = self._float("heading_alignment_tolerance_deg")
        self._local_recovery_step_m = self._float("local_recovery_step_m")
        self._local_recovery_retry_limit = self._int("local_recovery_retry_limit")
        self._move_timeout = self._float("move_timeout_sec")
        self._base_frame = self._str("base_frame")
        self._robot_radius_m = self._float("robot_radius_m")
        self._local_recovery_heading_offsets_deg = self._parse_float_list(
            self._str("local_recovery_heading_offsets_deg"),
            default=[15.0, 30.0, 45.0],
        )

        self._occupancy_map = OccupancyMap.from_yaml(self._map_yaml)
        builder = DeterministicTopologyBuilder(
            self._occupancy_map,
            waypoint_spacing_m=self._graph_waypoint_spacing_m,
            robot_radius_m=self._robot_radius_m,
            anchor_merge_distance_m=self._anchor_merge_distance_m,
        )
        self._topology_graph = builder.build(
            source_pose=(self._source_x, self._source_y),
            goal_pose=(self._dest_x, self._dest_y),
        )

        self._pose_lock = threading.Lock()
        self._odom_x = 0.0
        self._odom_y = 0.0
        self._odom_yaw = 0.0
        self._pose_ready = False
        self._alignment_ready = False
        self._map_to_odom: tuple[float, float, float] | None = None
        self._odom_frame = "odom"
        self._scan_lock = threading.Lock()
        self._latest_scan: LaserScan | None = None

        self._navigation_abort_reason: str | None = None
        self._replan_count = 0
        self._last_plan_notes = ""
        self._route_nodes: list[str] = []
        self._route_cursor = 0
        self._active_edge_id = ""
        self._active_edge_best_distance = math.inf
        self._active_edge_stall_steps = 0
        self._active_edge_waypoints: list[tuple[float, float]] = []
        self._active_edge_waypoint_index = 0
        self._active_edge_recovery_failures = 0
        self._failed_edges: dict[str, str] = {}
        self._controller_state = "WAITING_FOR_ODOM"
        self._workspace_dir = Path(self._map_yaml).resolve().parents[3]
        self._debug_image_dir = self._workspace_dir / "graph_step_debug" / datetime.now().strftime("%Y%m%d_%H%M%S")
        self._debug_image_dir.mkdir(parents=True, exist_ok=True)

        self._cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        status_topic = self._str("status_topic")
        self._status_pub = self.create_publisher(String, status_topic, 10)
        self._odom_sub = self.create_subscription(Odometry, "/odom", self._odom_cb, 10)
        self._scan_sub = self.create_subscription(LaserScan, "/scan", self._scan_cb, 10)

        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)
        self._static_tf_broadcaster = tf2_ros.StaticTransformBroadcaster(self)

        self._publish_status(
            "Graph-based LLM node started. "
            f"Built graph with {len(self._topology_graph.nodes)} nodes and {len(self._topology_graph.edges)} edges."
        )
        self._save_graph_debug_artifact()
        self._publish_status(f"Per-step debug images will be saved to: {self._debug_image_dir}")

        self._agent_thread = threading.Thread(target=self._run_agent, daemon=True)
        self._agent_thread.start()

    @property
    def map_yaml_path(self) -> str:
        return self._map_yaml

    def get_pose(self) -> tuple[float, float, float]:
        for base_frame in (self._base_frame, "base_link"):
            pose = self._lookup_map_pose_via_tf(base_frame)
            if pose is not None:
                return pose

        with self._pose_lock:
            if not self._alignment_ready or self._map_to_odom is None:
                raise RuntimeError("Map alignment is not ready; no valid map-frame pose is available.")
            return _compose_pose(
                self._map_to_odom,
                (self._odom_x, self._odom_y, self._odom_yaw),
            )

    def _save_graph_debug_artifact(self) -> None:
        graph_path = self._workspace_dir / "topology_graph_debug.json"
        graph_path.write_text(self._topology_graph.to_json(indent=2))

    def _save_step_debug_image(self, *, step: int, summary: str) -> None:
        try:
            x, y, yaw = self.get_pose()
        except Exception:
            return

        target_node_id = ""
        if self._route_nodes and 0 <= self._route_cursor < len(self._route_nodes):
            target_node_id = self._route_nodes[self._route_cursor]

        image = render_graph_debug_map(
            map_yaml_path=self._map_yaml,
            robot_x=x,
            robot_y=y,
            robot_yaw=yaw,
            graph=self._topology_graph,
            route_nodes=self._route_nodes,
            route_cursor=self._route_cursor,
            active_edge_id=self._active_edge_id,
            blocked_edge_ids=self._topology_graph.blocked_edge_ids(),
            source_x=self._source_x,
            source_y=self._source_y,
            dest_x=self._dest_x,
            dest_y=self._dest_y,
            caption_lines=[
                f"step={step} state={self._controller_state}",
                f"target_node={target_node_id or 'none'} active_edge={self._active_edge_id or 'none'}",
                f"route_cursor={self._route_cursor} route_len={len(self._route_nodes)} blocked_edges={len(self._topology_graph.blocked_edge_ids())}",
                f"pose=({x:.2f}, {y:.2f}) yaw={math.degrees(yaw):.1f} deg",
                f"summary={summary[:180]}",
            ],
        )
        image.save(self._debug_image_dir / f"step_{step:03d}_{self._controller_state.lower()}.png")
        image.save(self._debug_image_dir / "latest.png")

    def _current_graph_node_id(self) -> str:
        x, y, _ = self.get_pose()
        nearest = self._topology_graph.nearest_node(x, y)
        if nearest is None:
            raise RuntimeError("Could not localize the robot to the topology graph.")
        return nearest.node_id

    def _planning_context(self, reason: str) -> dict[str, Any]:
        x, y, yaw = self.get_pose()
        current_node_id = self._current_graph_node_id()
        deterministic_path = self._topology_graph.find_path(current_node_id, self._topology_graph.goal_node_id)
        return {
            "reason": reason,
            "pose": {
                "x": round(x, 3),
                "y": round(y, 3),
                "yaw_deg": round(math.degrees(yaw), 1),
            },
            "current_node_id": current_node_id,
            "goal_node_id": self._topology_graph.goal_node_id,
            "active_route_nodes": list(self._route_nodes),
            "active_route_cursor": self._route_cursor,
            "last_plan_notes": self._last_plan_notes,
            "blocked_edge_ids": self._topology_graph.blocked_edge_ids(),
            "failed_edges": dict(self._failed_edges),
            "deterministic_shortest_path": deterministic_path,
            "graph": self._topology_graph.to_compact_dict(),
        }

    def _plan_route(self, agent: Any, *, reason: str) -> str:
        current_node_id = self._current_graph_node_id()
        goal_node_id = self._topology_graph.goal_node_id
        decision = agent.plan(
            self._planning_context(reason),
            reason=reason,
        )

        candidate_path = list(decision.path_nodes)
        valid, message = self._topology_graph.validate_path(
            candidate_path,
            start_node_id=current_node_id,
            goal_node_id=goal_node_id,
        )
        if not valid:
            fallback_path = self._topology_graph.find_path(current_node_id, goal_node_id)
            if not fallback_path:
                self._controller_state = "FAIL"
                self._navigation_abort_reason = (
                    f"No route available from '{current_node_id}' to '{goal_node_id}' "
                    f"after graph updates. Last planner issue: {message}"
                )
                return self._navigation_abort_reason
            candidate_path = fallback_path
            decision.notes = (
                f"{decision.notes} | LLM route invalid ({message}); using deterministic shortest path."
            ).strip(" |")

        self._route_nodes = candidate_path
        self._route_cursor = 1 if len(candidate_path) > 1 else 0
        self._active_edge_id = ""
        self._active_edge_best_distance = math.inf
        self._active_edge_stall_steps = 0
        self._active_edge_waypoints = []
        self._active_edge_waypoint_index = 0
        self._active_edge_recovery_failures = 0
        self._last_plan_notes = decision.notes
        return (
            f"Planned node route: {self._route_nodes}. "
            f"Notes: {self._last_plan_notes or 'none'}"
        )

    def _scan_cb(self, msg: LaserScan) -> None:
        with self._scan_lock:
            self._latest_scan = msg

    def _laser_clearance_for_heading(
        self,
        target_heading: float,
        robot_yaw: float,
        *,
        window_deg: float = 10.0,
    ) -> float:
        with self._scan_lock:
            scan = self._latest_scan
        if scan is None or not scan.ranges:
            return math.inf

        relative_heading = math.atan2(
            math.sin(target_heading - robot_yaw),
            math.cos(target_heading - robot_yaw),
        )
        index_float = (relative_heading - scan.angle_min) / scan.angle_increment
        if not math.isfinite(index_float):
            return math.inf

        window_count = max(0, int(math.radians(window_deg) / max(scan.angle_increment, 1e-6)))
        center_index = int(round(index_float))
        valid_ranges: list[float] = []
        for idx in range(center_index - window_count, center_index + window_count + 1):
            if idx < 0 or idx >= len(scan.ranges):
                continue
            value = float(scan.ranges[idx])
            if not math.isfinite(value):
                continue
            if value < max(scan.range_min, 0.0) or value > scan.range_max:
                continue
            valid_ranges.append(value)
        if not valid_ranges:
            return math.inf
        return min(valid_ranges)

    def _oriented_edge_polyline(
        self,
        edge: Any,
        from_node_id: str,
        to_node_id: str,
    ) -> list[tuple[float, float]]:
        raw_polyline = list(edge.metadata.get("polyline", []))
        if not raw_polyline:
            from_node = self._topology_graph.get_node(from_node_id)
            to_node = self._topology_graph.get_node(to_node_id)
            if from_node is None or to_node is None:
                return []
            raw_polyline = [(from_node.x, from_node.y), (to_node.x, to_node.y)]

        polyline = [(float(point[0]), float(point[1])) for point in raw_polyline]
        if edge.from_node == from_node_id and edge.to_node == to_node_id:
            return polyline
        if edge.from_node == to_node_id and edge.to_node == from_node_id:
            return list(reversed(polyline))
        return polyline

    def _navigate_active_edge(self) -> NavigationStepResult:
        if not self._route_nodes:
            return NavigationStepResult("replan", "No active route. Triggering replanning.")
        if self._route_cursor >= len(self._route_nodes):
            return NavigationStepResult("replan", "Route exhausted before goal. Triggering replanning.")

        target_node_id = self._route_nodes[self._route_cursor]
        target_node = self._topology_graph.get_node(target_node_id)
        if target_node is None:
            return NavigationStepResult("replan", f"Target node '{target_node_id}' is missing. Triggering replanning.")

        current_node_id = self._route_nodes[self._route_cursor - 1]
        edge = self._topology_graph.edge_between(current_node_id, target_node_id)
        if edge is None:
            return NavigationStepResult(
                "replan",
                f"No graph edge exists between '{current_node_id}' and '{target_node_id}'.",
            )
        if edge.status == "blocked":
            return NavigationStepResult(
                "replan",
                f"Active edge '{edge.edge_id}' is blocked. Triggering replanning.",
            )

        if edge.edge_id != self._active_edge_id:
            oriented_polyline = self._oriented_edge_polyline(edge, current_node_id, target_node_id)
            self._active_edge_id = edge.edge_id
            self._active_edge_waypoints = oriented_polyline[1:] if len(oriented_polyline) > 1 else oriented_polyline
            self._active_edge_waypoint_index = 0
            self._active_edge_best_distance = math.inf
            self._active_edge_stall_steps = 0
            self._active_edge_recovery_failures = 0

        x, y, yaw = self.get_pose()
        distance_to_target = math.hypot(target_node.x - x, target_node.y - y)
        if distance_to_target <= self._node_reach_tolerance_m:
            self._route_cursor += 1
            self._active_edge_id = ""
            self._active_edge_best_distance = math.inf
            self._active_edge_stall_steps = 0
            self._active_edge_waypoints = []
            self._active_edge_waypoint_index = 0
            self._active_edge_recovery_failures = 0
            return NavigationStepResult(
                "reached",
                f"Reached node '{target_node_id}' at ({target_node.x:.2f}, {target_node.y:.2f}).",
            )

        while self._active_edge_waypoint_index < len(self._active_edge_waypoints):
            waypoint_x, waypoint_y = self._active_edge_waypoints[self._active_edge_waypoint_index]
            if math.hypot(waypoint_x - x, waypoint_y - y) > self._node_reach_tolerance_m:
                break
            self._active_edge_waypoint_index += 1
            self._active_edge_best_distance = math.inf
            self._active_edge_stall_steps = 0

        if self._active_edge_waypoint_index < len(self._active_edge_waypoints):
            guidance_x, guidance_y = self._active_edge_waypoints[self._active_edge_waypoint_index]
        else:
            guidance_x, guidance_y = target_node.x, target_node.y

        target_heading = math.atan2(guidance_y - y, guidance_x - x)
        heading_error_deg = math.degrees(
            math.atan2(math.sin(target_heading - yaw), math.cos(target_heading - yaw))
        )

        if abs(heading_error_deg) > self._heading_alignment_tolerance_deg:
            rotate_deg = max(-self._max_rotation_step_deg, min(self._max_rotation_step_deg, heading_error_deg))
            summary = self._rotate_step(rotate_deg)
            return NavigationStepResult(
                "rotating",
                (
                    f"Rotating toward node '{target_node_id}' "
                    f"(heading error {heading_error_deg:.1f} deg). {summary}"
                ),
            )

        distance_to_guidance = math.hypot(guidance_x - x, guidance_y - y)
        proposed_step = min(self._max_forward_step_m, distance_to_guidance)
        safe_step = self._safe_step_toward_heading(x, y, yaw, target_heading, proposed_step)
        if safe_step <= 0.05:
            recovery = self._run_local_edge_recovery(
                target_node_id=target_node_id,
                target_heading=target_heading,
                guidance_point=(guidance_x, guidance_y),
                distance_to_target=distance_to_target,
            )
            if recovery is not None:
                return recovery
            self._active_edge_recovery_failures += 1
            if self._active_edge_recovery_failures < self._local_recovery_retry_limit:
                return NavigationStepResult(
                    "moving",
                    (
                        f"Local recovery did not find a safe detour yet on edge '{edge.edge_id}'. "
                        f"Retrying edge traversal ({self._active_edge_recovery_failures}/{self._local_recovery_retry_limit - 1})."
                    ),
                )
            return NavigationStepResult(
                "blocked",
                (
                    f"Edge '{edge.edge_id}' toward node '{target_node_id}' is not locally traversable after "
                    f"{self._active_edge_recovery_failures} recovery attempts. Distance to target: {distance_to_target:.2f} m."
                ),
            )

        success, summary = self._forward_step(safe_step)
        if not success:
            return NavigationStepResult("blocked", summary)
        distance_after_move = math.hypot(guidance_x - self.get_pose()[0], guidance_y - self.get_pose()[1])
        if distance_after_move < self._active_edge_best_distance - self._edge_progress_epsilon_m:
            self._active_edge_best_distance = distance_after_move
            self._active_edge_stall_steps = 0
        else:
            self._active_edge_stall_steps += 1
            if self._active_edge_stall_steps >= self._edge_stall_step_limit:
                return NavigationStepResult(
                    "blocked",
                    (
                        f"Traversal of edge '{edge.edge_id}' stalled while approaching '{target_node_id}'. "
                        f"Best distance {self._active_edge_best_distance:.2f} m."
                    ),
                )
        self._active_edge_recovery_failures = 0
        return NavigationStepResult(
            "moving",
            f"Following edge '{edge.edge_id}' toward node '{target_node_id}'. {summary}",
        )

    def _safe_step_toward_heading(
        self,
        x: float,
        y: float,
        robot_yaw: float,
        heading: float,
        requested_step: float,
    ) -> float:
        step = min(requested_step, self._max_forward_step_m)
        laser_clearance = self._laser_clearance_for_heading(heading, robot_yaw)
        if math.isfinite(laser_clearance):
            step = min(step, max(0.0, laser_clearance - self._robot_radius_m - 0.05))
        while step > 0.05:
            end_x = x + step * math.cos(heading)
            end_y = y + step * math.sin(heading)
            if self._occupancy_map.is_segment_clear(
                (x, y),
                (end_x, end_y),
                robot_radius_m=self._robot_radius_m,
            ):
                return step
            step -= 0.05
        return 0.0

    def _run_local_edge_recovery(
        self,
        *,
        target_node_id: str,
        target_heading: float,
        guidance_point: tuple[float, float],
        distance_to_target: float,
    ) -> NavigationStepResult | None:
        x, y, yaw = self.get_pose()
        candidate_offsets = [0.0]
        for offset_deg in self._local_recovery_heading_offsets_deg:
            candidate_offsets.extend([-offset_deg, offset_deg])

        scored_candidates: list[tuple[float, float, float]] = []
        for offset_deg in candidate_offsets:
            heading = target_heading + math.radians(offset_deg)
            step = self._safe_step_toward_heading(
                x,
                y,
                yaw,
                heading,
                min(self._local_recovery_step_m, distance_to_target),
            )
            if step <= 0.05:
                continue
            clearance = self._laser_clearance_for_heading(heading, yaw)
            score = step + min(clearance, 2.0) * 0.1 - abs(offset_deg) * 0.002
            scored_candidates.append((score, heading, step))

        if not scored_candidates:
            return None

        scored_candidates.sort(reverse=True)
        _, best_heading, best_step = scored_candidates[0]
        heading_error_deg = math.degrees(
            math.atan2(math.sin(best_heading - yaw), math.cos(best_heading - yaw))
        )
        rotate_result = ""
        if abs(heading_error_deg) > self._heading_alignment_tolerance_deg:
            rotate_deg = max(-self._max_rotation_step_deg, min(self._max_rotation_step_deg, heading_error_deg))
            rotate_result = self._rotate_step(rotate_deg)

        success, move_result = self._forward_step(best_step)
        if not success:
            return None

        guidance_x, guidance_y = guidance_point
        self._active_edge_best_distance = math.hypot(guidance_x - self.get_pose()[0], guidance_y - self.get_pose()[1])
        self._active_edge_stall_steps = 0
        self._active_edge_recovery_failures = 0
        return NavigationStepResult(
            "moving",
            (
                f"Local planner deviated around a partial obstacle while staying on edge '{self._active_edge_id}' "
                f"toward node '{target_node_id}'. "
                f"{rotate_result + ' ' if rotate_result else ''}{move_result}"
            ).strip(),
        )

    def _mark_active_edge_blocked(self, reason: str) -> str:
        if not self._route_nodes or self._route_cursor <= 0 or self._route_cursor >= len(self._route_nodes):
            return "No active edge to block."
        from_node = self._route_nodes[self._route_cursor - 1]
        to_node = self._route_nodes[self._route_cursor]
        edge = self._topology_graph.edge_between(from_node, to_node)
        if edge is None:
            return f"Could not find an edge between '{from_node}' and '{to_node}' to block."
        edge.status = "blocked"
        edge.metadata["blocked_reason"] = reason
        self._failed_edges[edge.edge_id] = reason
        self._active_edge_id = ""
        self._active_edge_best_distance = math.inf
        self._active_edge_stall_steps = 0
        self._active_edge_waypoints = []
        self._active_edge_waypoint_index = 0
        self._active_edge_recovery_failures = 0
        return f"Marked edge '{edge.edge_id}' ({from_node} -> {to_node}) blocked: {reason}"

    def _forward_step(self, distance_m: float) -> tuple[bool, str]:
        x0, y0, yaw0 = self.get_pose()
        end_x = x0 + distance_m * math.cos(yaw0)
        end_y = y0 + distance_m * math.sin(yaw0)
        if not self._occupancy_map.is_segment_clear(
            (x0, y0),
            (end_x, end_y),
            robot_radius_m=self._robot_radius_m,
        ):
            return False, f"Forward step {distance_m:.2f} m rejected by occupancy validation."

        twist = Twist()
        twist.linear.x = min(abs(self._linear_speed), 0.26)

        t0 = time.monotonic()
        rate = self.create_rate(20)
        traveled = 0.0
        while traveled < distance_m:
            if time.monotonic() - t0 > self._move_timeout:
                self._stop()
                return False, f"Timed out after {traveled:.2f} m of requested {distance_m:.2f} m."
            self._cmd_pub.publish(twist)
            rate.sleep()
            cx, cy, _ = self.get_pose()
            traveled = math.hypot(cx - x0, cy - y0)

        self._stop()
        cx, cy, _ = self.get_pose()
        return True, f"Moved {traveled:.2f} m to ({cx:.2f}, {cy:.2f})."

    def _rotate_step(self, angle_deg: float) -> str:
        speed = min(abs(self._angular_speed), 1.0)
        angle_rad = math.radians(angle_deg)
        target_rad = abs(angle_rad)
        direction = 1.0 if angle_rad >= 0 else -1.0

        _, _, yaw0 = self.get_pose()
        twist = Twist()
        twist.angular.z = direction * speed

        t0 = time.monotonic()
        rate = self.create_rate(20)
        rotated = 0.0
        prev_yaw = yaw0

        while rotated < target_rad:
            if time.monotonic() - t0 > self._move_timeout:
                self._stop()
                return (
                    f"Timed out after rotating {math.degrees(rotated):.1f} deg "
                    f"of requested {angle_deg:.1f} deg."
                )
            self._cmd_pub.publish(twist)
            rate.sleep()

            _, _, yaw_now = self.get_pose()
            delta = yaw_now - prev_yaw
            if delta > math.pi:
                delta -= 2.0 * math.pi
            elif delta < -math.pi:
                delta += 2.0 * math.pi
            rotated += abs(delta)
            prev_yaw = yaw_now

        self._stop()
        _, _, final_yaw = self.get_pose()
        return f"Rotated {math.degrees(rotated):.1f} deg. Heading now {math.degrees(final_yaw):.1f} deg."

    def _stop(self) -> None:
        self._cmd_pub.publish(Twist())

    def _lookup_map_pose_via_tf(self, base_frame: str) -> tuple[float, float, float] | None:
        try:
            transform = self._tf_buffer.lookup_transform("map", base_frame, rclpy.time.Time())
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ):
            return None

        return (
            transform.transform.translation.x,
            transform.transform.translation.y,
            _yaw_from_quat(
                transform.transform.rotation.x,
                transform.transform.rotation.y,
                transform.transform.rotation.z,
                transform.transform.rotation.w,
            ),
        )

    def _publish_map_to_odom_transform(self) -> None:
        with self._pose_lock:
            if self._map_to_odom is None:
                return
            tx, ty, tyaw = self._map_to_odom
            odom_frame = self._odom_frame

        qx, qy, qz, qw = _quat_from_yaw(tyaw)
        transform = TransformStamped()
        transform.header.stamp = self.get_clock().now().to_msg()
        transform.header.frame_id = "map"
        transform.child_frame_id = odom_frame
        transform.transform.translation.x = tx
        transform.transform.translation.y = ty
        transform.transform.translation.z = 0.0
        transform.transform.rotation.x = qx
        transform.transform.rotation.y = qy
        transform.transform.rotation.z = qz
        transform.transform.rotation.w = qw
        self._static_tf_broadcaster.sendTransform(transform)

    def _odom_cb(self, msg: Odometry) -> None:
        pos = msg.pose.pose.position
        ori = msg.pose.pose.orientation
        yaw = _yaw_from_quat(ori.x, ori.y, ori.z, ori.w)
        publish_alignment = False

        with self._pose_lock:
            self._odom_x = pos.x
            self._odom_y = pos.y
            self._odom_yaw = yaw
            self._pose_ready = True
            self._odom_frame = msg.header.frame_id or "odom"
            if not self._alignment_ready:
                self._map_to_odom = _compose_pose(
                    (self._source_x, self._source_y, self._source_yaw),
                    _invert_pose((self._odom_x, self._odom_y, self._odom_yaw)),
                )
                self._alignment_ready = True
                publish_alignment = True

        if publish_alignment:
            self._publish_map_to_odom_transform()
            self._publish_status(
                "Aligned map->odom from the configured source pose and first odometry sample."
            )

    def _controller_iteration(self, agent: Any) -> str:
        x, y, _ = self.get_pose()
        distance_to_goal = math.hypot(self._dest_x - x, self._dest_y - y)
        if distance_to_goal <= self._goal_tolerance_m:
            self._controller_state = "GOAL_REACHED"
            return (
                f"GOAL REACHED. Current pose ({x:.2f}, {y:.2f}), "
                f"goal distance {distance_to_goal:.2f} m."
            )

        if not self._route_nodes or self._route_cursor >= len(self._route_nodes):
            self._controller_state = "REPLAN"
            summary = self._plan_route(agent, reason=f"plan_step_{self._replan_count}")
            if self._controller_state != "FAIL":
                self._controller_state = "FOLLOW_ROUTE"
            return summary

        result = self._navigate_active_edge()
        if result.status == "blocked":
            blocked_summary = self._mark_active_edge_blocked(result.message)
            self._replan_count += 1
            if self._replan_count > self._max_replans:
                self._controller_state = "FAIL"
                self._navigation_abort_reason = "Navigation failed after exhausting replans."
                return f"{blocked_summary} Replan budget exhausted."
            self._controller_state = "REPLAN"
            summary = self._plan_route(agent, reason=f"replan_after_{self._replan_count}")
            if self._controller_state != "FAIL":
                self._controller_state = "FOLLOW_ROUTE"
            return f"{blocked_summary} {summary}"
        if result.status == "replan":
            self._replan_count += 1
            if self._replan_count > self._max_replans:
                self._controller_state = "FAIL"
                self._navigation_abort_reason = "Navigation failed after exhausting replans."
                return result.message
            self._controller_state = "REPLAN"
            summary = self._plan_route(agent, reason=f"replan_after_{self._replan_count}")
            if self._controller_state != "FAIL":
                self._controller_state = "FOLLOW_ROUTE"
            return f"{result.message} {summary}"
        return result.message

    def _run_agent(self) -> None:
        self._publish_status("Waiting for odometry data...")
        while rclpy.ok():
            with self._pose_lock:
                if self._pose_ready and self._alignment_ready:
                    break
            time.sleep(0.2)

        self._controller_state = "READY"
        self._publish_status(
            f"Odometry aligned to map. Starting graph navigation from "
            f"({self._source_x:.2f}, {self._source_y:.2f}) to "
            f"({self._dest_x:.2f}, {self._dest_y:.2f})."
        )

        try:
            agent = build_agent()
        except Exception as exc:
            self._navigation_abort_reason = f"Failed to build LLM agent: {exc}"
            self._publish_status(self._navigation_abort_reason)
            self.get_logger().error(self._navigation_abort_reason)
            return

        logged_run_dir = False
        for step in range(1, self._max_agent_steps + 1):
            if not rclpy.ok():
                break
            if self._navigation_abort_reason is not None:
                self._publish_status(self._navigation_abort_reason)
                return

            self._publish_status(f"Controller step {step}/{self._max_agent_steps} [{self._controller_state}]")
            try:
                summary = self._controller_iteration(agent)
            except Exception as exc:
                self._navigation_abort_reason = f"Agent error on step {step}: {exc}"
                self._publish_status(self._navigation_abort_reason)
                self.get_logger().error(self._navigation_abort_reason)
                return

            self._publish_status(f"  -> {summary}")
            self._save_step_debug_image(step=step, summary=summary)
            if not logged_run_dir and agent.run_dir:
                self._publish_status(f"LLM debug artifacts saved to: {agent.run_dir}")
                logged_run_dir = True

            if self._controller_state == "GOAL_REACHED":
                return
            if self._controller_state == "FAIL":
                if self._navigation_abort_reason is None:
                    self._navigation_abort_reason = "Navigation failed after exhausting replans."
                self._publish_status(self._navigation_abort_reason)
                return

        x, y, _ = self.get_pose()
        distance_to_goal = math.hypot(self._dest_x - x, self._dest_y - y)
        self._publish_status(
            f"Controller exhausted {self._max_agent_steps} steps. "
            f"Distance to goal: {distance_to_goal:.2f} m."
        )

    def _publish_status(self, message: str) -> None:
        msg = String()
        msg.data = message
        self._status_pub.publish(msg)
        self.get_logger().info(message)

    def _float(self, name: str) -> float:
        return self.get_parameter(name).get_parameter_value().double_value

    def _int(self, name: str) -> int:
        return self.get_parameter(name).get_parameter_value().integer_value

    def _str(self, name: str) -> str:
        return self.get_parameter(name).get_parameter_value().string_value

    def _parse_float_list(self, raw: str, *, default: list[float]) -> list[float]:
        values: list[float] = []
        for token in raw.split(","):
            token = token.strip()
            if not token:
                continue
            try:
                values.append(float(token))
            except ValueError:
                continue
        return values or list(default)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = LlmAgentNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
