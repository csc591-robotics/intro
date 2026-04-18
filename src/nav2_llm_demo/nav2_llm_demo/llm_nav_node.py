"""LangChain-backed decision layer that routes a robot through Nav2."""

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from geometry_msgs.msg import PoseStamped
from nav2_simple_commander.robot_navigator import BasicNavigator
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from .llm import load_route_graph_from_map_poses, plan_route


def _load_world_to_map_offset(sidecar_path: str) -> tuple[float, float]:
    """Read (offset_x, offset_y) from a .world_map.yaml sidecar.

    Returns (0.0, 0.0) if the path is empty or the file cannot be read,
    preserving backward compatibility with hand-crafted maps that have no
    sidecar (for those, world frame == map frame so the offset is zero).
    """
    if not sidecar_path:
        return 0.0, 0.0
    p = Path(sidecar_path).expanduser()
    if not p.is_file():
        # Try resolving relative to the workspace root (next to install/)
        ws = Path(__file__).resolve().parents[4]
        p = ws / sidecar_path
    if not p.is_file():
        return 0.0, 0.0
    try:
        import yaml  # available in any ROS 2 Humble Python env
        with p.open() as fh:
            data = yaml.safe_load(fh)
        off = data.get('world_to_map_offset', [0.0, 0.0])
        return float(off[0]), float(off[1])
    except Exception:  # noqa: BLE001
        return 0.0, 0.0


@dataclass
class SegmentState:
    """Execution state for the current route segment."""

    from_checkpoint: str
    to_checkpoint: str
    start_time: float
    last_progress_time: float
    best_distance: float | None = None


class LlmNavNode(Node):
    """Mission controller that asks the LLM to choose route segments."""

    def __init__(self) -> None:
        """Initialize ROS interfaces, parameters, and navigation state."""
        super().__init__('llm_nav_node')

        self.declare_parameter('request_topic', '/navigation_request')
        self.declare_parameter('status_topic', '/navigation_status')
        self.declare_parameter('active_goal_pose_topic', '/active_goal_pose')
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('navigation_timeout_sec', 90.0)
        self.declare_parameter('stall_timeout_sec', 15.0)
        self.declare_parameter('stall_min_progress_m', 0.15)
        self.declare_parameter('max_replans', 10)
        self.declare_parameter('max_decision_attempts', 2)
        self.declare_parameter('map_poses_path', '')
        self.declare_parameter('map_name', '')
        self.declare_parameter(
            'planner_notes',
            (
                'You are the high-level routing layer. Choose among allowed '
                'checkpoints and edges only. Nav2 handles obstacle avoidance '
                'and local path planning. Never use blocked edges. If a route '
                'fails, choose a different legal route from the current '
                'checkpoint.'
            ),
        )

        # ── Load route graph from map_poses.yaml ─────────────────────────────
        map_poses_path = self._string_param('map_poses_path')
        map_name = self._string_param('map_name')
        if not map_poses_path or not map_name:
            raise RuntimeError(
                'map_poses_path and map_name parameters are required. '
                'Pass them to llm_nav.launch.py:\n'
                '  map_poses_path:=/workspace/intro/src/custom_map_builder/maps/map_poses.yaml\n'
                '  map_name:=warehouse'
            )

        self.get_logger().info(
            f'Loading route graph from map_poses.yaml: '
            f'{map_poses_path}  map={map_name}'
        )
        self._graph, sidecar_path = load_route_graph_from_map_poses(
            map_poses_path, map_name
        )

        # ── World→map offset (from sidecar embedded in map_poses.yaml) ───────
        self._map_offset_x, self._map_offset_y = _load_world_to_map_offset(
            sidecar_path
        )
        if sidecar_path:
            self.get_logger().info(
                f'sidecar={sidecar_path!r} → '
                f'world→map offset=({self._map_offset_x:.4f}, '
                f'{self._map_offset_y:.4f}). '
                'Checkpoint coords are in Gazebo world frame and will be '
                'shifted by this offset before being sent to Nav2.'
            )
        else:
            self.get_logger().info(
                'No sidecar in map_poses.yaml — checkpoint coords sent to '
                'Nav2 unchanged (world frame == map frame).'
            )

        # graph - navigation environment representation
        self._checkpoints = self._graph['checkpoints']
        self._edges = {
            (edge['from'], edge['to']) for edge in self._graph['edges']
        }
        self._adjacency = {name: [] for name in self._checkpoints}
        for from_node, to_node in self._edges:
            self._adjacency[from_node].append(to_node)
        self._goal_aliases = self._graph['goal_aliases']
        self._start_checkpoint = self._graph['start_checkpoint']
        self._current_checkpoint = self._start_checkpoint

        request_topic = self._string_param('request_topic')
        status_topic = self._string_param('status_topic')
        goal_pose_topic = self._string_param('active_goal_pose_topic')

        self._status_pub = self.create_publisher(String, status_topic, 10)
        self._goal_pub = self.create_publisher(PoseStamped, goal_pose_topic, 10)
        self._request_sub = self.create_subscription(
            String,
            request_topic,
            self._handle_request,
            10,
        )

        self._navigator = BasicNavigator()
        self._mission_goal_request = ''
        self._blocked_edges: set[tuple[str, str]] = set()
        self._last_failure_reason = ''
        self._replan_count = 0
        self._current_goal_alias = ''
        self._active_route: list[str] = []
        self._route_index = 0
        self._segment: SegmentState | None = None
        self._mission_timer = self.create_timer(1.0, self._mission_tick)

        self.get_logger().info(
            f'Loaded route graph with {len(self._checkpoints)} checkpoints and '
            f'{len(self._edges)} directed edges'
        )
        # We do NOT run AMCL — Gazebo's diff_drive plugin + our static
        # map→odom TF are the localization. BasicNavigator defaults to
        # waiting for the 'amcl' lifecycle node, which would hang here
        # forever. Pointing the localizer wait at 'controller_server'
        # makes BasicNavigator skip both the AMCL check and the
        # /initialpose wait while still confirming bt_navigator is up.
        self._publish_status('Waiting for Nav2 to become active')
        self._navigator.waitUntilNav2Active(localizer='controller_server')
        self._publish_status(
            'Nav2 active. Waiting for high-level mission requests'
        )

    def _handle_request(self, msg: String) -> None:
        """Accept a new mission request if no mission is currently active."""
        goal_request = msg.data.strip()
        if not goal_request:
            self.get_logger().warning('Ignoring empty mission request')
            return

        if self._mission_goal_request:
            self._publish_status(
                'Mission already in progress; ignoring new request'
            )
            return

        self._mission_goal_request = goal_request
        self._blocked_edges = set()
        self._last_failure_reason = ''
        self._replan_count = 0
        self._current_goal_alias = ''
        self._active_route = []
        self._route_index = 0
        self._segment = None
        self._publish_status(f'Mission requested: {goal_request}')

    def _mission_tick(self) -> None:
        """Advance the active mission without running concurrent mission code."""
        if not self._mission_goal_request:
            return

        try:
            if not self._active_route:

                # plan route with llm
                goal_alias, route = plan_route(
                    goal_request=self._mission_goal_request,
                    blocked_edges=self._blocked_edges,
                    failure_reason=self._last_failure_reason,
                    current_checkpoint=self._current_checkpoint,
                    goal_aliases=self._goal_aliases,
                    checkpoints=self._checkpoints,
                    edges=self._edges,
                    planner_notes=self._string_param('planner_notes'),
                    max_attempts=self._int_param('max_decision_attempts'),
                    publish_status=self._publish_status,
                )
                self._current_goal_alias = goal_alias
                self._active_route = route
                self._route_index = 1
                self._publish_status(
                    f"LLM chose route to '{self._current_goal_alias}': "
                    f"{' -> '.join(route)}"
                )
                return

            if self._segment is not None:
                self._tick_active_segment()
                return

            if self._route_index >= len(self._active_route):
                self._publish_status(
                    "Mission complete. Reached goal alias "
                    f"'{self._current_goal_alias}'"
                )
                self._reset_mission_state()
                return

            self._start_next_segment()
        except Exception as exc:
            self._publish_status(f'Mission failed: {exc}')
            self.get_logger().error(f'Mission failed: {exc}')
            self._reset_mission_state()

    def _start_next_segment(self) -> None:
        """Send the next checkpoint goal to Nav2 and initialize segment tracking."""
        next_checkpoint = self._active_route[self._route_index]
        checkpoint = self._checkpoints[next_checkpoint]

        pose = PoseStamped()
        pose.header.frame_id = self._string_param('map_frame')
        pose.header.stamp = self.get_clock().now().to_msg()
        # Checkpoint coords are stored in Gazebo world frame.
        # Apply world→map offset so Nav2 receives map-frame coordinates.
        pose.pose.position.x = float(checkpoint['x']) + self._map_offset_x
        pose.pose.position.y = float(checkpoint['y']) + self._map_offset_y
        pose.pose.position.z = 0.0

        half_yaw = float(checkpoint['yaw']) / 2.0
        pose.pose.orientation.z = math.sin(half_yaw)
        pose.pose.orientation.w = math.cos(half_yaw)

        self._goal_pub.publish(pose)
        self._publish_status(
            f'Executing segment {self._current_checkpoint} -> '
            f'{next_checkpoint}'
        )
        self._navigator.goToPose(pose)
        now = time.monotonic()
        self._segment = SegmentState(
            from_checkpoint=self._current_checkpoint,
            to_checkpoint=next_checkpoint,
            start_time=now,
            last_progress_time=now,
        )

    def _tick_active_segment(self) -> None:
        """Check the in-flight Nav2 segment and replan or advance as needed."""
        outcome = self._monitor_active_segment()
        if outcome is None:
            return

        segment = self._segment
        if segment is None:
            return
        self._segment = None

        if outcome['status'] == 'success':
            self._current_checkpoint = segment.to_checkpoint
            self._route_index += 1
            self._publish_status(
                f"Reached checkpoint '{self._current_checkpoint}'"
            )
            return

        if self._replan_count >= self._int_param('max_replans'):
            raise RuntimeError(
                'Mission failed after exhausting replans: '
                f"{outcome['reason']}"
            )

        failed_edge = outcome.get('failed_edge')
        if failed_edge is not None:
            self._blocked_edges.add(failed_edge)

        self._replan_count += 1
        self._last_failure_reason = outcome['reason']
        self._active_route = []
        self._route_index = 0
        self._publish_status(
            f'Replanning after route failure: {self._last_failure_reason}'
        )

    def _monitor_active_segment(self) -> dict[str, Any] | None:
        """Monitor the active Nav2 segment until it succeeds or fails."""
        segment = self._segment
        if segment is None:
            return None

        navigation_timeout_sec = self._float_param('navigation_timeout_sec')
        stall_timeout_sec = self._float_param('stall_timeout_sec')
        stall_min_progress_m = self._float_param('stall_min_progress_m')

        now = time.monotonic()
        elapsed = now - segment.start_time
        if elapsed > navigation_timeout_sec:
            self._navigator.cancelTask()
            return {
                'status': 'failed',
                'reason': (
                    f'Segment {segment.from_checkpoint}->'
                    f'{segment.to_checkpoint} timed out after '
                    f'{navigation_timeout_sec:.1f}s'
                ),
                'failed_edge': (
                    segment.from_checkpoint,
                    segment.to_checkpoint,
                ),
            }

        if not self._navigator.isTaskComplete():
            feedback = self._navigator.getFeedback()
            distance_remaining = getattr(feedback, 'distance_remaining', None)
            if distance_remaining is not None:
                distance_remaining = float(distance_remaining)
                if segment.best_distance is None:
                    segment.best_distance = distance_remaining
                    segment.last_progress_time = now
                elif (
                    segment.best_distance - distance_remaining
                    >= stall_min_progress_m
                ):
                    segment.best_distance = distance_remaining
                    segment.last_progress_time = now
                elif (
                    now - segment.last_progress_time
                    > stall_timeout_sec
                ):
                    self._navigator.cancelTask()
                    return {
                        'status': 'failed',
                        'reason': (
                            f'Segment {segment.from_checkpoint}->'
                            f'{segment.to_checkpoint} stalled for '
                            f'{stall_timeout_sec:.1f}s'
                        ),
                        'failed_edge': (
                            segment.from_checkpoint,
                            segment.to_checkpoint,
                        ),
                    }

            return None

        result = str(self._navigator.getResult())
        if 'SUCCEEDED' in result:
            return {'status': 'success'}

        return {
            'status': 'failed',
            'reason': (
                f'Segment {segment.from_checkpoint}->'
                f'{segment.to_checkpoint} returned '
                f'Nav2 result {result}'
            ),
            'failed_edge': (
                segment.from_checkpoint,
                segment.to_checkpoint,
            ),
        }

    def _reset_mission_state(self) -> None:
        """Clear mission execution state so the next request can be accepted."""
        self._mission_goal_request = ''
        self._blocked_edges = set()
        self._last_failure_reason = ''
        self._replan_count = 0
        self._current_goal_alias = ''
        self._active_route = []
        self._route_index = 0
        self._segment = None

    def _publish_status(self, message: str) -> None:
        """Publish a status update and mirror it to the node logger."""
        status = String()
        status.data = message
        self._status_pub.publish(status)
        self.get_logger().info(message)

    def _string_param(self, name: str) -> str:
        """Return a string ROS parameter value."""
        return self.get_parameter(name).get_parameter_value().string_value

    def _float_param(self, name: str) -> float:
        """Return a floating-point ROS parameter value."""
        return self.get_parameter(name).get_parameter_value().double_value

    def _int_param(self, name: str) -> int:
        """Return an integer ROS parameter value."""
        return self.get_parameter(name).get_parameter_value().integer_value


def main(args: list[str] | None = None) -> None:
    """Initialize ROS, run the node, and cleanly shut down."""
    rclpy.init(args=args)
    node = LlmNavNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
