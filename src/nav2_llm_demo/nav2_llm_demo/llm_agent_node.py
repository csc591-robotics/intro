"""ROS 2 node that runs a vision-based LLM agent to navigate a robot.

The node:
- subscribes to /odom for robot position tracking
- publishes to /cmd_vel for direct robot control
- aligns map -> odom from the configured source pose and the first odom sample
- uses TF2 when available, otherwise uses the aligned odom pose in the map frame
- runs a VisionNavigationAgent that replans only on meaningful navigation events
"""

import math
import threading
import time
from pathlib import Path
from typing import Any

from geometry_msgs.msg import TransformStamped, Twist
from nav_msgs.msg import Odometry
import numpy as np
from PIL import Image
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import tf2_ros
import yaml

from .llm.llm_agent import build_agent, set_controller


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


class LlmAgentNode(Node):
    """Vision-based LLM planner with deterministic local motion control."""

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
        self.declare_parameter("max_agent_steps", 50)
        self.declare_parameter("linear_speed", 0.15)
        self.declare_parameter("angular_speed", 0.5)
        self.declare_parameter("status_topic", "/navigation_status")
        self.declare_parameter("move_timeout_sec", 30.0)
        self.declare_parameter("base_frame", "base_footprint")

        self._source_x = self._float("source_x")
        self._source_y = self._float("source_y")
        self._source_yaw = self._float("source_yaw")
        self._dest_x = self._float("dest_x")
        self._dest_y = self._float("dest_y")
        self._map_yaml = self._str("map_yaml_path")
        self._linear_speed = self._float("linear_speed")
        self._angular_speed = self._float("angular_speed")
        self._move_timeout = self._float("move_timeout_sec")
        self._base_frame = self._str("base_frame")
        self._map_resolution, self._map_origin_x, self._map_origin_y, self._map_grid = self._load_map()
        self._max_clearance_probe_m = 1.2
        self._robot_radius_m = 0.11
        self._last_progress_m: float | None = None
        self._blocked_forward_streak = 0
        self._navigation_abort_reason: str | None = None
        self._controller_state = "IDLE"
        self._controller_step = 0
        self._replan_count = 0
        self._recovery_count = 0
        self._max_replans = 3
        self._max_recovery_attempts = 3
        self._recovery_attempts_current_event = 0
        self._current_route_intent = "continue_route"
        self._route_plan_queue: list[str] = []
        self._route_intent_pending_alignment = False
        self._branch_entry_move_pending = False
        self._branch_entry_target_side = "none"
        self._failed_route_intents: list[str] = []
        self._route_failure_streak = 0
        self._last_route_failure_reason = ""
        self._last_plan_notes = ""

        self._pose_lock = threading.Lock()
        self._odom_x = 0.0
        self._odom_y = 0.0
        self._odom_yaw = 0.0
        self._pose_ready = False
        self._alignment_ready = False
        self._map_to_odom: tuple[float, float, float] | None = None
        self._odom_frame = "odom"

        self._cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        status_topic = self._str("status_topic")
        self._status_pub = self.create_publisher(String, status_topic, 10)

        self._odom_sub = self.create_subscription(
            Odometry, "/odom", self._odom_cb, 10,
        )

        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)
        self._static_tf_broadcaster = tf2_ros.StaticTransformBroadcaster(self)

        set_controller(self)

        self._publish_status("LLM agent node started. Waiting for odometry...")

        self._agent_thread = threading.Thread(target=self._run_agent, daemon=True)
        self._agent_thread.start()

    # ------------------------------------------------------------------
    # RobotController protocol implementation
    # ------------------------------------------------------------------

    @property
    def map_yaml_path(self) -> str:
        return self._map_yaml

    @property
    def source_x(self) -> float:
        return self._source_x

    @property
    def source_y(self) -> float:
        return self._source_y

    @property
    def dest_x(self) -> float:
        return self._dest_x

    @property
    def dest_y(self) -> float:
        return self._dest_y

    def get_pose(self) -> tuple[float, float, float]:
        """Return (x, y, yaw) in the map frame."""
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

    def get_navigation_context(self) -> dict[str, float | bool | str | None]:
        x, y, yaw = self.get_pose()
        distance_to_goal = math.hypot(self._dest_x - x, self._dest_y - y)
        goal_heading = math.atan2(self._dest_y - y, self._dest_x - x)
        heading_error = math.degrees(
            math.atan2(math.sin(goal_heading - yaw), math.cos(goal_heading - yaw))
        )
        forward_free_space = self._clearance_in_direction(x, y, yaw)
        left_free_space = self._clearance_in_direction(x, y, yaw + math.pi / 2.0)
        right_free_space = self._clearance_in_direction(x, y, yaw - math.pi / 2.0)
        recommended_step = self._recommended_step(
            forward_free_space,
            distance_to_goal,
        )
        return {
            "distance_to_goal_m": round(distance_to_goal, 3),
            "heading_error_deg": round(heading_error, 1),
            "forward_clearance_m": round(forward_free_space, 3),
            "left_clearance_m": round(left_free_space, 3),
            "right_clearance_m": round(right_free_space, 3),
            "recommended_step_m": round(recommended_step, 3),
            "last_progress_m": None if self._last_progress_m is None else round(self._last_progress_m, 3),
            "blocked_streak": self._blocked_forward_streak,
            "in_known_free_space": self._is_known_free(x, y),
        }

    def turn_toward_goal(self) -> str:
        """Rotate toward the goal using deterministic heading logic."""
        x, y, yaw = self.get_pose()
        heading_to_goal = math.atan2(self._dest_y - y, self._dest_x - x)
        heading_error_deg = math.degrees(
            math.atan2(math.sin(heading_to_goal - yaw), math.cos(heading_to_goal - yaw))
        )

        if abs(heading_error_deg) < 5.0:
            return f"Already roughly facing the goal. Heading error {heading_error_deg:.1f} deg."

        turn_deg = max(-60.0, min(60.0, heading_error_deg))
        return self.rotate(turn_deg)

    def advance_step(self) -> str:
        """Advance with controller-chosen step sizing."""
        return self.move_forward(1.0)

    def recover_to_open_side(self) -> str:
        """Rotate toward the more open side without forcing an immediate forward move."""
        x, y, yaw = self.get_pose()
        left_free_space = self._clearance_in_direction(x, y, yaw + math.pi / 2.0)
        right_free_space = self._clearance_in_direction(x, y, yaw - math.pi / 2.0)

        if left_free_space == right_free_space:
            turn_deg = 35.0
            chosen_side = "left"
        elif left_free_space > right_free_space:
            turn_deg = 35.0
            chosen_side = "left"
        else:
            turn_deg = -35.0
            chosen_side = "right"

        rotate_result = self.rotate(turn_deg)
        return (
            f"Recovery chose the {chosen_side} side "
            f"(left {left_free_space:.2f} m, right {right_free_space:.2f} m). "
            f"{rotate_result}"
        )

    def back_up(self) -> str:
        """Back up by a short deterministic amount."""
        return self.move_forward(-0.25)

    def _planning_context(self, reason: str) -> dict[str, Any]:
        x, y, yaw = self.get_pose()
        base_context = self.get_navigation_context()
        return {
            "reason": reason,
            "controller_state": self._controller_state,
            "controller_step": self._controller_step,
            "last_plan_notes": self._last_plan_notes,
            "pose": {
                "x": round(x, 3),
                "y": round(y, 3),
                "yaw_deg": round(math.degrees(yaw), 1),
            },
            "navigation_context": base_context,
            "route_memory": {
                "current_route_intent": self._current_route_intent,
                "route_plan_queue": list(self._route_plan_queue),
                "failed_route_intents": list(self._failed_route_intents),
                "route_failure_streak": self._route_failure_streak,
                "last_route_failure_reason": self._last_route_failure_reason,
            },
        }

    def _choose_recovery_side(self, ctx: dict[str, Any]) -> str:
        x, y, yaw = self.get_pose()
        left_viable = self._direction_viable(x, y, yaw + math.pi / 2.0, 0.25)
        right_viable = self._direction_viable(x, y, yaw - math.pi / 2.0, 0.25)
        if left_viable and not right_viable:
            return "left"
        if right_viable and not left_viable:
            return "right"
        left_free = float(ctx["left_clearance_m"])
        right_free = float(ctx["right_clearance_m"])
        return "left" if left_free >= right_free else "right"

    def _recover_with_side(self, side: str, ctx: dict[str, Any]) -> str:
        left_free = float(ctx["left_clearance_m"])
        right_free = float(ctx["right_clearance_m"])
        turn_deg = 35.0 if side == "left" else -35.0
        rotate_result = self.rotate(turn_deg)
        return (
            f"Recovery chose the {side} side "
            f"(left {left_free:.2f} m, right {right_free:.2f} m). {rotate_result}"
        )

    def _advance_succeeded(self, result: str) -> bool:
        return result.startswith("Moved ")

    def _clear_branch_entry_intent(self) -> None:
        """Clear any short-lived branch entry intent after it has been executed."""
        self._current_route_intent = self._route_plan_queue.pop(0) if self._route_plan_queue else "continue_route"
        self._route_intent_pending_alignment = False
        self._branch_entry_move_pending = False
        self._branch_entry_target_side = "none"

    def _record_route_failure(self, reason: str) -> None:
        self._route_failure_streak += 1
        self._last_route_failure_reason = reason
        if self._current_route_intent in {"take_left_branch", "take_right_branch"}:
            if self._current_route_intent not in self._failed_route_intents:
                self._failed_route_intents.append(self._current_route_intent)

    def _run_recovery_sequence(self) -> tuple[str, str]:
        ctx = self.get_navigation_context()
        side = self._choose_recovery_side(ctx)
        rotate_summary = self._recover_with_side(side, ctx)

        ctx_after_turn = self.get_navigation_context()
        if float(ctx_after_turn["recommended_step_m"]) > 0.0:
            advance_result = self.advance_step()
            if self._advance_succeeded(advance_result):
                return "NAVIGATE", f"{rotate_summary} | {advance_result}"

        back_up_result = self.back_up()
        if self._recovery_attempts_current_event < self._max_recovery_attempts:
            return (
                "BLOCKED",
                f"{rotate_summary} | {back_up_result} | "
                f"Recovery attempt {self._recovery_attempts_current_event}/{self._max_recovery_attempts} did not restore a safe route.",
            )
        self._record_route_failure("bounded local recovery exhausted")
        return (
            "REPLAN",
            f"{rotate_summary} | {back_up_result} | "
            f"Route failed after {self._recovery_attempts_current_event} recovery attempts.",
        )

    def move_forward(self, distance_m: float, speed: float | None = None) -> str:
        """Drive forward/backward by *distance_m* meters using cmd_vel."""
        if speed is None:
            speed = self._linear_speed
        speed = min(abs(speed), 0.26)

        x0, y0, yaw0 = self.get_pose()
        goal_before = math.hypot(self._dest_x - x0, self._dest_y - y0)
        forward_free_space = self._clearance_in_direction(
            x0, y0, yaw0 if distance_m >= 0 else yaw0 + math.pi
        )
        requested_dist = abs(distance_m)
        target_dist = requested_dist
        direction = 1.0 if distance_m >= 0 else -1.0
        if direction > 0:
            safe_dist = self._recommended_step(
                forward_free_space,
                goal_before,
            )
            if safe_dist <= 0.05:
                self._blocked_forward_streak += 1
                blocked_msg = (
                    f"Advance blocked: free space ahead {forward_free_space:.2f} m is too small for a safe step. "
                    f"Consider recover_to_open_side() or back_up(). "
                    f"Blocked streak: {self._blocked_forward_streak}."
                )
                return blocked_msg
            target_dist = safe_dist
        else:
            target_dist = min(requested_dist, 0.4)

        if not self._segment_is_clear(x0, y0, yaw0, direction * target_dist):
            self._blocked_forward_streak += 1
            return (
                f"Advance blocked: path validation rejected {target_dist:.2f} m because the robot footprint "
                f"would enter occupied space. Blocked streak: {self._blocked_forward_streak}."
            )

        twist = Twist()
        twist.linear.x = direction * speed

        t0 = time.monotonic()
        rate = self.create_rate(20)
        traveled = 0.0

        while traveled < target_dist:
            if time.monotonic() - t0 > self._move_timeout:
                self._stop()
                return f"Timed out after moving {traveled:.2f} m of requested {distance_m:.2f} m."

            self._cmd_pub.publish(twist)
            rate.sleep()

            cx, cy, _ = self.get_pose()
            traveled = math.hypot(cx - x0, cy - y0)

        self._stop()
        cx, cy, _ = self.get_pose()
        goal_after = math.hypot(self._dest_x - cx, self._dest_y - cy)
        progress = goal_before - goal_after
        self._last_progress_m = progress
        self._blocked_forward_streak = 0
        self._recovery_attempts_current_event = 0
        details = []
        # Report the controller-selected step size for this move.
        details.append(f"step {target_dist:.2f} m")
        # If a reverse request was too large, report the capped distance we actually allowed.
        if direction < 0 and target_dist + 1e-6 < requested_dist:
            details.append(f"Rejected distance: {requested_dist:.2f} m to {target_dist:.2f} m")
        # Only report progress when it is meaningfully positive; moving away can be part of recovery.
        if progress > 0.05:
            details.append(f"progress {progress:.2f} m")
        suffix = f" ({'; '.join(details)})" if details else ""
        return (
            f"Moved {traveled:.2f} m. Now at ({cx:.2f}, {cy:.2f}). "
            f"Goal distance: {goal_after:.2f} m.{suffix}"
        )

    def rotate(self, angle_deg: float, speed: float | None = None) -> str:
        """Rotate in place by *angle_deg* degrees."""
        if speed is None:
            speed = self._angular_speed
        speed = min(abs(speed), 1.0)

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

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _stop(self) -> None:
        self._cmd_pub.publish(Twist())

    def _load_map(self) -> tuple[float, float, float, np.ndarray]:
        map_yaml = Path(self._map_yaml)
        with map_yaml.open() as f:
            meta = yaml.safe_load(f)

        resolution = float(meta["resolution"])
        origin = meta.get("origin", [0.0, 0.0, 0.0])
        origin_x = float(origin[0])
        origin_y = float(origin[1])
        grid = np.array(Image.open(map_yaml.parent / meta["image"]).convert("L"))
        if int(meta.get("negate", 0)):
            grid = 255 - grid
        return resolution, origin_x, origin_y, grid

    def _world_to_grid(self, wx: float, wy: float) -> tuple[int, int]:
        gx = int((wx - self._map_origin_x) / self._map_resolution)
        gy = self._map_grid.shape[0] - 1 - int((wy - self._map_origin_y) / self._map_resolution)
        return gx, gy

    def _grid_value(self, wx: float, wy: float) -> int | None:
        gx, gy = self._world_to_grid(wx, wy)
        if gy < 0 or gy >= self._map_grid.shape[0] or gx < 0 or gx >= self._map_grid.shape[1]:
            return None
        return int(self._map_grid[gy, gx])

    def _is_known_free(self, wx: float, wy: float) -> bool:
        value = self._grid_value(wx, wy)
        return value is not None and value >= 200

    def _footprint_is_clear(self, wx: float, wy: float) -> bool:
        """Conservative occupancy check for the robot footprint."""
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
                    wx + self._robot_radius_m * math.cos(angle),
                    wy + self._robot_radius_m * math.sin(angle),
                )
            )
        for sx, sy in sample_points:
            value = self._grid_value(sx, sy)
            if value is None or value < 200:
                return False
        return True

    def _segment_is_clear(self, wx: float, wy: float, yaw: float, signed_distance_m: float) -> bool:
        """Check the entire forward or reverse segment against occupancy."""
        step = max(self._map_resolution / 2.0, 0.03)
        total = abs(signed_distance_m)
        direction = 1.0 if signed_distance_m >= 0.0 else -1.0
        traveled = 0.0
        while traveled <= total + 1e-6:
            sample_x = wx + direction * traveled * math.cos(yaw)
            sample_y = wy + direction * traveled * math.sin(yaw)
            if not self._footprint_is_clear(sample_x, sample_y):
                return False
            traveled += step
        end_x = wx + signed_distance_m * math.cos(yaw)
        end_y = wy + signed_distance_m * math.sin(yaw)
        return self._footprint_is_clear(end_x, end_y)

    def _direction_viable(self, wx: float, wy: float, yaw: float, probe_distance_m: float) -> bool:
        """Use the same footprint-aware collision check for branch viability as for motion."""
        return self._segment_is_clear(wx, wy, yaw, probe_distance_m)

    def _clearance_in_direction(self, wx: float, wy: float, yaw: float) -> float:
        step = max(self._map_resolution, 0.05)
        distance = 0.0
        while distance <= self._max_clearance_probe_m:
            sample_x = wx + distance * math.cos(yaw)
            sample_y = wy + distance * math.sin(yaw)
            value = self._grid_value(sample_x, sample_y)
            if value is None or value < 200:
                break
            distance += step
        return max(0.0, distance - step)

    def _recommended_step(
        self,
        forward_free_space: float,
        distance_to_goal: float,
    ) -> float:
        # Reserve a small safety margin instead of using all visible free space ahead.
        safe_forward_room = max(0.0, forward_free_space - 0.10)
        if safe_forward_room <= 0.05:
            return 0.0

        # If the remaining safe room is tiny, only allow a tiny step.
        if safe_forward_room < 0.15:
            step_m = 0.08
        elif distance_to_goal <= 0.75:
            step_m = min(distance_to_goal, safe_forward_room, 0.20)
        elif distance_to_goal <= 2.0:
            step_m = min(distance_to_goal, safe_forward_room, 0.35)
        else:
            step_m = min(safe_forward_room, 0.50)

        return step_m

    def _lookup_map_pose_via_tf(self, base_frame: str) -> tuple[float, float, float] | None:
        try:
            t = self._tf_buffer.lookup_transform("map", base_frame, rclpy.time.Time())
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ):
            return None

        return (
            t.transform.translation.x,
            t.transform.translation.y,
            _yaw_from_quat(
                t.transform.rotation.x,
                t.transform.rotation.y,
                t.transform.rotation.z,
                t.transform.rotation.w,
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

    # ------------------------------------------------------------------
    # Event-driven navigation controller
    # ------------------------------------------------------------------

    def _apply_planner_decision(self, decision: Any) -> str:
        sequence = list(decision.route_sequence)
        self._current_route_intent = sequence[0] if sequence else "continue_route"
        self._route_plan_queue = sequence[1:] if len(sequence) > 1 else []
        self._route_intent_pending_alignment = self._current_route_intent in {
            "take_left_branch",
            "take_right_branch",
        }
        self._branch_entry_move_pending = self._route_intent_pending_alignment
        self._branch_entry_target_side = (
            "left" if self._current_route_intent == "take_left_branch"
            else "right" if self._current_route_intent == "take_right_branch"
            else "none"
        )
        self._last_plan_notes = str(decision.notes)
        self._recovery_attempts_current_event = 0
        return (
            f"Planner decision: route_sequence={[self._current_route_intent, *self._route_plan_queue]}, "
            f"notes={self._last_plan_notes}"
        )

    def _plan_with_agent(self, agent: Any, *, reason: str, include_map: bool = True) -> str:
        decision = agent.plan(
            self._planning_context(reason),
            reason=reason,
            include_map=include_map,
        )
        if reason.startswith("replan"):
            self._replan_count += 1
        return self._apply_planner_decision(decision)

    def _controller_iteration(self, agent: Any) -> str:
        x, y, _ = self.get_pose()
        dist = math.hypot(self._dest_x - x, self._dest_y - y)

        if self._controller_state == "NAVIGATE":
            ctx = self.get_navigation_context()
            if self._current_route_intent == "continue_route" and self._route_plan_queue:
                self._current_route_intent = self._route_plan_queue.pop(0)
                self._route_intent_pending_alignment = self._current_route_intent in {
                    "take_left_branch",
                    "take_right_branch",
                }
                self._branch_entry_move_pending = self._route_intent_pending_alignment
                self._branch_entry_target_side = (
                    "left" if self._current_route_intent == "take_left_branch"
                    else "right" if self._current_route_intent == "take_right_branch"
                    else "none"
                )
                return f"Advancing to next planned action: {self._current_route_intent}"
            if self._current_route_intent == "backtrack":
                back_up_result = self.back_up()
                if self._advance_succeeded(back_up_result):
                    self._current_route_intent = self._route_plan_queue.pop(0) if self._route_plan_queue else "continue_route"
                    return f"Executing backtrack intent. {back_up_result}"
                self._controller_state = "REPLAN"
                self._current_route_intent = "continue_route"
                self._route_plan_queue = []
                return f"Executing backtrack intent. {back_up_result}"
            if self._route_intent_pending_alignment:
                x, y, yaw = self.get_pose()
                left_viable = self._direction_viable(x, y, yaw + math.pi / 2.0, 0.25)
                right_viable = self._direction_viable(x, y, yaw - math.pi / 2.0, 0.25)
                if self._current_route_intent == "take_left_branch" and left_viable:
                    self._route_intent_pending_alignment = False
                    return f"Aligning to planned left branch. {self.rotate(35.0)}"
                if self._current_route_intent == "take_right_branch" and right_viable:
                    self._route_intent_pending_alignment = False
                    return f"Aligning to planned right branch. {self.rotate(-35.0)}"
                failed_branch = self._current_route_intent
                self._record_route_failure(f"{failed_branch} is not locally traversable")
                self._route_plan_queue = []
                self._current_route_intent = "continue_route"
                self._route_intent_pending_alignment = False
                self._branch_entry_move_pending = False
                self._branch_entry_target_side = "none"
                self._controller_state = "REPLAN"
                return f"Planned branch {failed_branch} is not locally traversable. Triggering replanning."
            if float(ctx["recommended_step_m"]) > 0.0:
                if (
                    self._current_route_intent == "continue_route"
                    and abs(float(ctx["heading_error_deg"])) > 25.0
                    and float(ctx["forward_clearance_m"]) > 0.20
                ):
                    result = self.turn_toward_goal()
                    self._blocked_forward_streak = 0
                    return result
                advance_result = self.advance_step()
                if self._advance_succeeded(advance_result):
                    if self._branch_entry_move_pending:
                        self._clear_branch_entry_intent()
                        return f"{advance_result} Branch entry complete; resuming normal navigation."
                    return advance_result
                self._controller_state = "BLOCKED"
                return f"{advance_result} Transitioning to deterministic recovery."
            self._controller_state = "BLOCKED"
            return (
                f"Navigation blocked: forward clearance {float(ctx['forward_clearance_m']):.2f} m, "
                f"recommended step {float(ctx['recommended_step_m']):.2f} m."
            )

        if self._controller_state == "BLOCKED":
            if self._recovery_attempts_current_event == 0:
                self._route_failure_streak = 0
            self._controller_state = "RECOVER"
            return "Blocked event recorded. Starting deterministic recovery."

        if self._controller_state == "RECOVER":
            self._recovery_count += 1
            self._recovery_attempts_current_event += 1
            next_state, summary = self._run_recovery_sequence()
            self._controller_state = next_state
            return summary

        if self._controller_state == "REPLAN":
            if self._replan_count >= self._max_replans:
                self._controller_state = "FAIL"
                return "Replan budget exhausted."
            summary = self._plan_with_agent(
                agent,
                reason=f"replan_after_route_failure_{self._controller_step}",
                include_map=True,
            )
            self._controller_state = "NAVIGATE"
            self._blocked_forward_streak = 0
            return summary

        if self._controller_state == "FAIL":
            self._navigation_abort_reason = "Navigation failed after exhausting replans."
            return self._navigation_abort_reason

        return f"Unhandled controller state: {self._controller_state}"

    def _run_agent(self) -> None:
        """Wait for odom alignment, build the planner, and run the state machine."""
        self.get_logger().info("Waiting for odometry data...")
        while rclpy.ok():
            with self._pose_lock:
                if self._pose_ready and self._alignment_ready:
                    break
            time.sleep(0.2)

        self._publish_status(
            f"Odometry aligned to map. Starting navigation from "
            f"({self._source_x:.2f}, {self._source_y:.2f}) to "
            f"({self._dest_x:.2f}, {self._dest_y:.2f})"
        )

        try:
            agent = build_agent()
        except Exception as exc:
            self._publish_status(f"Failed to build LLM agent: {exc}")
            self.get_logger().error(f"Agent build failed: {exc}")
            return

        agent.initialize(
            self._source_x, self._source_y,
            self._dest_x, self._dest_y,
        )

        max_steps = self._int("max_agent_steps")
        goal_tol = self._float("goal_tolerance_m")
        logged_run_dir = False
        self._controller_state = "REPLAN"
        self._replan_count = 0

        for step in range(1, max_steps + 1):
            self._controller_step = step
            if not rclpy.ok():
                break

            x, y, _ = self.get_pose()
            dist = math.hypot(self._dest_x - x, self._dest_y - y)
            if dist < goal_tol:
                self._controller_state = "GOAL_REACHED"
                self._publish_status(
                    f"GOAL REACHED after {step - 1} controller steps! "
                    f"Distance: {dist:.2f} m (tolerance: {goal_tol:.2f} m)"
                )
                return

            self._publish_status(f"Controller step {step}/{max_steps} [{self._controller_state}]")

            try:
                summary = self._controller_iteration(agent)
                self._publish_status(f"  -> {summary}")
                if not logged_run_dir and agent.run_dir:
                    self._publish_status(f"Debug images saved to: {agent.run_dir}")
                    logged_run_dir = True
            except Exception as exc:
                self._publish_status(f"Agent error on step {step}: {exc}")
                self.get_logger().error(f"Agent step {step} failed: {exc}")
                break

            if self._navigation_abort_reason is not None:
                self._publish_status(self._navigation_abort_reason)
                return
            if self._controller_state == "FAIL":
                self._navigation_abort_reason = "Navigation failed after exhausting replans."
                self._publish_status(self._navigation_abort_reason)
                return

        x, y, _ = self.get_pose()
        dist = math.hypot(self._dest_x - x, self._dest_y - y)
        self._publish_status(
            f"Controller exhausted {max_steps} steps. "
            f"Distance to goal: {dist:.2f} m."
        )


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = LlmAgentNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
