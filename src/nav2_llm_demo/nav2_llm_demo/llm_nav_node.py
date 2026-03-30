"""Groq-backed decision layer that routes a robot through Nav2."""

import json
import math
import os
import threading
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
        self.declare_parameter('max_replans', 3)
        self.declare_parameter('route_graph_path', '')
        self.declare_parameter(
            'planner_notes',
            (
                'You are the high-level routing layer. Choose among allowed '
                'checkpoints and edges only. Nav2 handles obstacle avoidance '
                'and local path planning.'
            ),
        )

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
        self._mission_lock = threading.Lock()
        self._mission_active = False
        self._mission_thread: threading.Thread | None = None

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
        goal_request = msg.data.strip()
        if not goal_request:
            self.get_logger().warning('Ignoring empty mission request')
            return

        with self._mission_lock:
            if self._mission_active:
                self._publish_status(
                    'Mission already in progress; ignoring new request'
                )
                return
            self._mission_active = True

        self._mission_thread = threading.Thread(
            target=self._run_mission,
            args=(goal_request,),
            daemon=True,
        )
        self._mission_thread.start()

    def _run_mission(self, goal_request: str) -> None:
        blocked_edges: set[tuple[str, str]] = set()
        last_failure_reason = ''
        replan_count = 0

        try:
            self._publish_status(f'Mission requested: {goal_request}')
            while True:
                decision = self._query_groq_for_route(
                    goal_request=goal_request,
                    blocked_edges=blocked_edges,
                    failure_reason=last_failure_reason,
                )
                route = self._validate_route_decision(
                    decision,
                    blocked_edges,
                )
                goal_alias = decision['goal_alias']

                self._publish_status(
                    f"Groq chose route to '{goal_alias}': {' -> '.join(route)}"
                )
                outcome = self._execute_route(route)
                if outcome['status'] == 'success':
                    self._publish_status(
                        f"Mission complete. Reached goal alias '{goal_alias}'"
                    )
                    return

                if replan_count >= self._int_param('max_replans'):
                    raise RuntimeError(
                        'Mission failed after exhausting replans: '
                        f"{outcome['reason']}"
                    )

                failed_edge = outcome.get('failed_edge')
                if failed_edge is not None:
                    blocked_edges.add(failed_edge)

                replan_count += 1
                last_failure_reason = outcome['reason']
                self._publish_status(
                    f'Replanning after route failure: {last_failure_reason}'
                )
        except Exception as exc:
            self._publish_status(f'Mission failed: {exc}')
            self.get_logger().error(f'Mission failed: {exc}')
        finally:
            with self._mission_lock:
                self._mission_active = False

    def _execute_route(self, route: list[str]) -> dict[str, Any]:
        for next_checkpoint in route[1:]:
            pose = self._pose_for_checkpoint(next_checkpoint)
            self._goal_pub.publish(pose)
            self._publish_status(
                f'Executing segment {self._current_checkpoint} -> '
                f'{next_checkpoint}'
            )
            self._navigator.goToPose(pose)

            outcome = self._wait_for_segment_result(
                self._current_checkpoint,
                next_checkpoint,
            )
            if outcome['status'] != 'success':
                return outcome

            self._current_checkpoint = next_checkpoint
            self._publish_status(
                f"Reached checkpoint '{self._current_checkpoint}'"
            )

        return {'status': 'success'}

    def _wait_for_segment_result(
        self,
        from_checkpoint: str,
        to_checkpoint: str,
    ) -> dict[str, Any]:
        navigation_timeout_sec = self._float_param('navigation_timeout_sec')
        stall_timeout_sec = self._float_param('stall_timeout_sec')
        stall_min_progress_m = self._float_param('stall_min_progress_m')

        start_time = time.monotonic()
        last_progress_time = start_time
        best_distance: float | None = None

        while not self._navigator.isTaskComplete():
            elapsed = time.monotonic() - start_time
            if elapsed > navigation_timeout_sec:
                self._navigator.cancelTask()
                return {
                    'status': 'failed',
                    'reason': (
                        f'Segment {from_checkpoint}->{to_checkpoint} '
                        f'timed out after {navigation_timeout_sec:.1f}s'
                    ),
                    'failed_edge': (from_checkpoint, to_checkpoint),
                }

            feedback = self._navigator.getFeedback()
            distance_remaining = getattr(feedback, 'distance_remaining', None)
            if distance_remaining is not None:
                distance_remaining = float(distance_remaining)
                if best_distance is None:
                    best_distance = distance_remaining
                    last_progress_time = time.monotonic()
                elif best_distance - distance_remaining >= stall_min_progress_m:
                    best_distance = distance_remaining
                    last_progress_time = time.monotonic()
                elif time.monotonic() - last_progress_time > stall_timeout_sec:
                    self._navigator.cancelTask()
                    return {
                        'status': 'failed',
                        'reason': (
                            f'Segment {from_checkpoint}->{to_checkpoint} '
                            f'stalled for {stall_timeout_sec:.1f}s'
                        ),
                        'failed_edge': (from_checkpoint, to_checkpoint),
                    }

            time.sleep(1.0)

        result = str(self._navigator.getResult())
        if 'SUCCEEDED' in result:
            return {'status': 'success'}

        return {
            'status': 'failed',
            'reason': (
                f'Segment {from_checkpoint}->{to_checkpoint} returned '
                f'Nav2 result {result}'
            ),
            'failed_edge': (from_checkpoint, to_checkpoint),
        }

    def _query_groq_for_route(
        self,
        goal_request: str,
        blocked_edges: set[tuple[str, str]],
        failure_reason: str,
    ) -> dict[str, Any]:
        api_key = self._get_api_key()
        payload = {
            'model': self._string_param('groq_model'),
            'temperature': 0.2,
            'messages': [
                {
                    'role': 'system',
                    'content': self._build_system_prompt(),
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

    def _validate_route_decision(
        self,
        decision: dict[str, Any],
        blocked_edges: set[tuple[str, str]],
    ) -> list[str]:
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

    def _build_system_prompt(self) -> str:
        planner_notes = self._string_param('planner_notes')
        return (
            'You are the high-level route decision layer for a TurtleBot. '
            'Do not create arbitrary coordinates. '
            'Choose only from the checkpoint graph provided in the user JSON. '
            'Output JSON only with this schema: '
            '{"goal_alias": string, "route": [string, ...], "reason": string}. '
            'The route must begin at current_checkpoint, end at a checkpoint '
            'allowed by the chosen goal_alias, and only use allowed edges that '
            'are not blocked. '
            f'{planner_notes}'
        )

    def _build_decision_context(
        self,
        goal_request: str,
        blocked_edges: set[tuple[str, str]],
        failure_reason: str,
    ) -> dict[str, Any]:
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
        adjacency = {name: [] for name in self._checkpoints}
        for from_node, to_node in self._edges:
            adjacency[from_node].append(to_node)
        return adjacency

    def _pose_for_checkpoint(self, checkpoint_name: str) -> PoseStamped:
        checkpoint = self._checkpoints[checkpoint_name]

        pose = PoseStamped()
        pose.header.frame_id = self._string_param('map_frame')
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = float(checkpoint['x'])
        pose.pose.position.y = float(checkpoint['y'])
        pose.pose.position.z = 0.0

        half_yaw = float(checkpoint['yaw']) / 2.0
        pose.pose.orientation.z = math.sin(half_yaw)
        pose.pose.orientation.w = math.cos(half_yaw)
        return pose

    def _get_api_key(self) -> str:
        api_key = self._string_param('groq_api_key') or os.environ.get(
            'GROQ_API_KEY',
            '',
        )
        if not api_key:
            raise RuntimeError(
                'Missing Groq API key. Set the groq_api_key parameter or '
                'GROQ_API_KEY environment variable.'
            )
        return api_key

    def _publish_status(self, message: str) -> None:
        status = String()
        status.data = message
        self._status_pub.publish(status)
        self.get_logger().info(message)

    def _string_param(self, name: str) -> str:
        return self.get_parameter(name).get_parameter_value().string_value

    def _float_param(self, name: str) -> float:
        return self.get_parameter(name).get_parameter_value().double_value

    def _int_param(self, name: str) -> int:
        return self.get_parameter(name).get_parameter_value().integer_value


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = LlmNavNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
