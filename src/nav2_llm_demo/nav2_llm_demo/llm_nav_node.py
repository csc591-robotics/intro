"""Groq-backed decision layer that routes a robot through Nav2."""

import json
import math
import os
import time
from pathlib import Path
from typing import Any
from urllib import error, request

from geometry_msgs.msg import PoseStamped
from nav2_simple_commander.robot_navigator import BasicNavigator
import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class LlmNavNode(Node):
    """Mission controller that asks Groq to choose route segments."""

    def __init__(self) -> None:
        """Initialize ROS interfaces, parameters, and navigation state."""
        super().__init__('llm_nav_node')

        self.declare_parameter('groq_api_key', '')
        self.declare_parameter('groq_model', 'llama-3.3-70b-versatile')
        self.declare_parameter('request_topic', '/navigation_request')
        self.declare_parameter('status_topic', '/navigation_status')
        self.declare_parameter('active_goal_pose_topic', '/active_goal_pose')
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('navigation_timeout_sec', 90.0)
        self.declare_parameter('stall_timeout_sec', 15.0)
        self.declare_parameter('stall_min_progress_m', 0.15)
        self.declare_parameter('max_replans', 10)
        self.declare_parameter('max_decision_attempts', 2)
        self.declare_parameter('route_graph_path', '')
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

        # graph - navigation environment representation
        self._graph = self._load_route_graph()
        
        self._checkpoints = self._graph['checkpoints']
        self._edges = {
            (edge['from'], edge['to']) for edge in self._graph['edges']
        }
        self._adjacency = self._build_adjacency()
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
        self._mission_active = False
        self._mission_goal_request = ''
        self._blocked_edges: set[tuple[str, str]] = set()
        self._last_failure_reason = ''
        self._replan_count = 0
        self._current_goal_alias = ''
        self._active_route: list[str] = []
        self._route_index = 0
        self._segment_from_checkpoint = ''
        self._segment_to_checkpoint = ''
        self._segment_start_time = 0.0
        self._segment_last_progress_time = 0.0
        self._segment_best_distance: float | None = None
        self._segment_active = False
        self._mission_timer = self.create_timer(1.0, self._mission_tick)

        self.get_logger().info(
            f'Loaded route graph with {len(self._checkpoints)} checkpoints and '
            f'{len(self._edges)} directed edges'
        )
        self._publish_status('Waiting for Nav2 to become active')
        self._navigator.waitUntilNav2Active()
        self._publish_status(
            'Nav2 active. Waiting for high-level mission requests'
        )

    def _handle_request(self, msg: String) -> None:
        """Accept a new mission request if no mission is currently active."""
        goal_request = msg.data.strip()
        if not goal_request:
            self.get_logger().warning('Ignoring empty mission request')
            return

        if self._mission_active:
            self._publish_status(
                'Mission already in progress; ignoring new request'
            )
            return

        self._mission_active = True
        self._mission_goal_request = goal_request
        self._blocked_edges = set()
        self._last_failure_reason = ''
        self._replan_count = 0
        self._current_goal_alias = ''
        self._active_route = []
        self._route_index = 0
        self._segment_from_checkpoint = ''
        self._segment_to_checkpoint = ''
        self._segment_start_time = 0.0
        self._segment_last_progress_time = 0.0
        self._segment_best_distance = None
        self._segment_active = False
        self._publish_status(f'Mission requested: {goal_request}')

    def _mission_tick(self) -> None:
        """Advance the active mission without running concurrent mission code."""
        if not self._mission_active:
            return

        try:
            if not self._active_route:
                self._plan_route()
                return

            if self._segment_active:
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

    def _plan_route(self) -> None:
        """Ask the LLM for a route and store it for segment-by-segment execution."""
        max_attempts = self._int_param('max_decision_attempts')
        last_error = 'No route decision attempts were made'

        # attempt to make a valid decision x amount of times before gigivn up
        for attempt in range(1, max_attempts + 1):
            decision = self._make_decision(
                goal_request=self._mission_goal_request,
                blocked_edges=self._blocked_edges,
                failure_reason=self._last_failure_reason,
            )
            try:
                route = self._validate_decision(decision, self._blocked_edges)
            except RuntimeError as exc:
                last_error = str(exc)
                self._publish_status(
                    f'Ignoring invalid route from Groq on attempt {attempt}/'
                    f'{max_attempts}: {last_error}'
                )
                continue

            self._current_goal_alias = decision['goal_alias']
            self._active_route = route
            self._route_index = 1
            self._publish_status(
                f"Groq chose route to '{self._current_goal_alias}': "
                f"{' -> '.join(route)}"
            )
            return

        raise RuntimeError(
            'Groq did not return a valid legal route after '
            f'{max_attempts} attempts: {last_error}'
        )

    def _start_next_segment(self) -> None:
        """Send the next checkpoint goal to Nav2 and initialize segment tracking."""
        next_checkpoint = self._active_route[self._route_index]
        checkpoint = self._checkpoints[next_checkpoint]

        pose = PoseStamped()
        pose.header.frame_id = self._string_param('map_frame')
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = float(checkpoint['x'])
        pose.pose.position.y = float(checkpoint['y'])
        pose.pose.position.z = 0.0

        half_yaw = float(checkpoint['yaw']) / 2.0
        pose.pose.orientation.z = math.sin(half_yaw)
        pose.pose.orientation.w = math.cos(half_yaw)

        self._goal_pub.publish(pose)
        self._segment_from_checkpoint = self._current_checkpoint
        self._segment_to_checkpoint = next_checkpoint
        self._publish_status(
            f'Executing segment {self._segment_from_checkpoint} -> '
            f'{self._segment_to_checkpoint}'
        )
        self._navigator.goToPose(pose)
        now = time.monotonic()
        self._segment_start_time = now
        self._segment_last_progress_time = now
        self._segment_best_distance = None
        self._segment_active = True

    def _tick_active_segment(self) -> None:
        """Check the in-flight Nav2 segment and replan or advance as needed."""
        outcome = self._monitor_active_segment()
        if outcome is None:
            return

        self._segment_active = False

        if outcome['status'] == 'success':
            self._current_checkpoint = self._segment_to_checkpoint
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
        navigation_timeout_sec = self._float_param('navigation_timeout_sec')
        stall_timeout_sec = self._float_param('stall_timeout_sec')
        stall_min_progress_m = self._float_param('stall_min_progress_m')

        elapsed = time.monotonic() - self._segment_start_time
        if elapsed > navigation_timeout_sec:
            self._navigator.cancelTask()
            return {
                'status': 'failed',
                'reason': (
                    f'Segment {self._segment_from_checkpoint}->'
                    f'{self._segment_to_checkpoint} timed out after '
                    f'{navigation_timeout_sec:.1f}s'
                ),
                'failed_edge': (
                    self._segment_from_checkpoint,
                    self._segment_to_checkpoint,
                ),
            }

        if not self._navigator.isTaskComplete():
            feedback = self._navigator.getFeedback()
            distance_remaining = getattr(feedback, 'distance_remaining', None)
            if distance_remaining is not None:
                distance_remaining = float(distance_remaining)
                if self._segment_best_distance is None:
                    self._segment_best_distance = distance_remaining
                    self._segment_last_progress_time = time.monotonic()
                elif (
                    self._segment_best_distance - distance_remaining
                    >= stall_min_progress_m
                ):
                    self._segment_best_distance = distance_remaining
                    self._segment_last_progress_time = time.monotonic()
                elif (
                    time.monotonic() - self._segment_last_progress_time
                    > stall_timeout_sec
                ):
                    self._navigator.cancelTask()
                    return {
                        'status': 'failed',
                        'reason': (
                            f'Segment {self._segment_from_checkpoint}->'
                            f'{self._segment_to_checkpoint} stalled for '
                            f'{stall_timeout_sec:.1f}s'
                        ),
                        'failed_edge': (
                            self._segment_from_checkpoint,
                            self._segment_to_checkpoint,
                        ),
                    }

            return None

        result = str(self._navigator.getResult())
        if 'SUCCEEDED' in result:
            return {'status': 'success'}

        return {
            'status': 'failed',
            'reason': (
                f'Segment {self._segment_from_checkpoint}->'
                f'{self._segment_to_checkpoint} returned '
                f'Nav2 result {result}'
            ),
            'failed_edge': (
                self._segment_from_checkpoint,
                self._segment_to_checkpoint,
            ),
        }

    def _reset_mission_state(self) -> None:
        """Clear mission execution state so the next request can be accepted."""
        self._mission_active = False
        self._mission_goal_request = ''
        self._blocked_edges = set()
        self._last_failure_reason = ''
        self._replan_count = 0
        self._current_goal_alias = ''
        self._active_route = []
        self._route_index = 0
        self._segment_from_checkpoint = ''
        self._segment_to_checkpoint = ''
        self._segment_start_time = 0.0
        self._segment_last_progress_time = 0.0
        self._segment_best_distance = None
        self._segment_active = False

    def _make_decision(
        self,
        goal_request: str,
        blocked_edges: set[tuple[str, str]],
        failure_reason: str,
    ) -> dict[str, Any]:
        """Ask the LLM to make the next route decision."""
        api_key = self._string_param('groq_api_key') or os.environ.get(
            'GROQ_API_KEY',
            '',
        )
        if not api_key:
            raise RuntimeError(
                'Missing Groq API key. Set the groq_api_key parameter or '
                'GROQ_API_KEY environment variable.'
            )
        planner_notes = self._string_param('planner_notes')
        payload = {
            'model': self._string_param('groq_model'),
            'temperature': 0.2,
            'messages': [
                {
                    'role': 'system',
                    'content': (
                        'You are the high-level route decision layer for a '
                        'TurtleBot. '
                        'Do not create arbitrary coordinates. '
                        'Choose only from the checkpoint graph provided in '
                        'the user JSON. '
                        'Output JSON only with this schema: '
                        '{"goal_alias": string, "route": [string, ...], '
                        '"reason": string}. '
                        'The route must begin at current_checkpoint, end at a '
                        'checkpoint allowed by the chosen goal_alias, and '
                        'only use allowed edges that are not blocked. '
                        f'{planner_notes}'
                    ),
                },
                {
                    'role': 'user',
                    'content': json.dumps(
                        self._build_decision_context(
                            goal_request,
                            blocked_edges,
                            failure_reason,
                        )
                    ),
                },
            ],
            'response_format': {'type': 'json_object'},
        }

        body = json.dumps(payload).encode('utf-8')
        http_request = request.Request(
            'https://api.groq.com/openai/v1/chat/completions',
            data=body,
            headers={
                'Authorization': f'Bearer {api_key}',
                'Content-Type': 'application/json',
                'User-Agent': 'nav2-llm-demo/0.1 (+groq api client)',
            },
            method='POST',
        )

        try:
            with request.urlopen(http_request, timeout=30) as response:
                response_body = response.read().decode('utf-8')
        except error.HTTPError as exc:
            details = exc.read().decode('utf-8', errors='replace')
            raise RuntimeError(
                f'Groq API returned HTTP {exc.code}: {details}'
            ) from exc
        except error.URLError as exc:
            raise RuntimeError(f'Could not reach Groq API: {exc.reason}') from exc

        response_json = json.loads(response_body)
        content = response_json['choices'][0]['message']['content']
        decision = json.loads(content)

        if 'goal_alias' not in decision or 'route' not in decision:
            raise RuntimeError(
                'Groq response must include goal_alias and route fields'
            )

        if not isinstance(decision['route'], list):
            raise RuntimeError('Groq route must be a JSON array')

        decision.setdefault('reason', '')
        return decision

    def _validate_decision(
        self,
        decision: dict[str, Any],
        blocked_edges: set[tuple[str, str]],
    ) -> list[str]:
        """Validate that the LLM decision stays within the allowed graph."""
        goal_alias = decision['goal_alias']
        if goal_alias not in self._goal_aliases:
            raise RuntimeError(f"Unknown goal alias '{goal_alias}'")

        route = [str(node) for node in decision['route']]
        if len(route) < 2:
            raise RuntimeError('Route must include at least start and goal')

        if route[0] != self._current_checkpoint:
            raise RuntimeError(
                'Route must start at the current checkpoint '
                f"'{self._current_checkpoint}'"
            )

        allowed_goals = set(self._goal_aliases[goal_alias])
        if route[-1] not in allowed_goals:
            raise RuntimeError(
                f"Route for goal alias '{goal_alias}' must end at one of "
                f'{sorted(allowed_goals)}'
            )

        for node in route:
            if node not in self._checkpoints:
                raise RuntimeError(f"Unknown checkpoint '{node}' in route")

        for from_node, to_node in zip(route, route[1:]):
            if (from_node, to_node) not in self._edges:
                raise RuntimeError(
                    f"Route includes invalid edge '{from_node}->{to_node}'"
                )
            if (from_node, to_node) in blocked_edges:
                raise RuntimeError(
                    f"Route reuses blocked edge '{from_node}->{to_node}'"
                )

        return route

    def _build_decision_context(
        self,
        goal_request: str,
        blocked_edges: set[tuple[str, str]],
        failure_reason: str,
    ) -> dict[str, Any]:
        """Assemble the structured graph context for route selection."""
        return {
            'goal_request': goal_request,
            'current_checkpoint': self._current_checkpoint,
            'goal_aliases': self._goal_aliases,
            'checkpoint_descriptions': {
                name: checkpoint['description']
                for name, checkpoint in self._checkpoints.items()
            },
            'allowed_edges': [
                {'from': from_node, 'to': to_node}
                for from_node, to_node in sorted(self._edges)
                if (from_node, to_node) not in blocked_edges
            ],
            'blocked_edges': [
                {'from': from_node, 'to': to_node}
                for from_node, to_node in sorted(blocked_edges)
            ],
            'last_failure_reason': failure_reason,
        }

    def _load_route_graph(self) -> dict[str, Any]:
        """Load and validate the route graph JSON from disk."""
        graph_path = self._string_param('route_graph_path')
        if not graph_path:
            raise RuntimeError('route_graph_path parameter is required')

        path = Path(graph_path)
        if not path.is_file():
            raise RuntimeError(f'Route graph file not found: {graph_path}')

        with path.open('r', encoding='utf-8') as handle:
            graph = json.load(handle)

        required_keys = {
            'start_checkpoint',
            'checkpoints',
            'edges',
            'goal_aliases',
        }
        missing = required_keys - set(graph.keys())
        if missing:
            raise RuntimeError(
                f'Route graph missing required keys: {sorted(missing)}'
            )

        checkpoints = graph['checkpoints']
        if graph['start_checkpoint'] not in checkpoints:
            raise RuntimeError('start_checkpoint must exist in checkpoints')

        for name, checkpoint in checkpoints.items():
            for key in ('x', 'y', 'yaw', 'description'):
                if key not in checkpoint:
                    raise RuntimeError(
                        f"Checkpoint '{name}' missing required key '{key}'"
                    )

        for edge in graph['edges']:
            from_node = edge.get('from')
            to_node = edge.get('to')
            if from_node not in checkpoints or to_node not in checkpoints:
                raise RuntimeError(
                    f'Edge references unknown checkpoint: {edge}'
                )

        for alias, goals in graph['goal_aliases'].items():
            if not goals:
                raise RuntimeError(f"Goal alias '{alias}' has no targets")
            for goal in goals:
                if goal not in checkpoints:
                    raise RuntimeError(
                        f"Goal alias '{alias}' references unknown checkpoint "
                        f"'{goal}'"
                    )

        return graph

    def _build_adjacency(self) -> dict[str, list[str]]:
        """Build a simple outgoing-edge map for each checkpoint."""
        adjacency = {name: [] for name in self._checkpoints}
        for from_node, to_node in self._edges:
            adjacency[from_node].append(to_node)
        return adjacency

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
