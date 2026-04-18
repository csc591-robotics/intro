"""ROS 2 node that runs a vision-based LLM agent to navigate a robot.

The node:
- subscribes to /odom for robot position tracking
- publishes to /cmd_vel for direct robot control
- aligns map -> odom from the configured source pose and the first odom sample
- uses TF2 when available, otherwise uses the aligned odom pose in the map frame
- runs a VisionNavigationAgent that sees an annotated map and calls tools
"""

import math
import threading
import time
from pathlib import Path

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
    """Vision-based LLM agent that directly controls the robot."""

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
        self._last_progress_m: float | None = None
        self._regression_streak = 0

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
        forward_clearance = self._clearance_in_direction(x, y, yaw)
        left_clearance = self._clearance_in_direction(x, y, yaw + math.pi / 2.0)
        right_clearance = self._clearance_in_direction(x, y, yaw - math.pi / 2.0)
        recommended_step = self._recommended_step(
            forward_clearance,
            abs(heading_error),
            distance_to_goal,
            self._regression_streak,
        )
        return {
            "distance_to_goal_m": round(distance_to_goal, 3),
            "heading_error_deg": round(heading_error, 1),
            "forward_clearance_m": round(forward_clearance, 3),
            "left_clearance_m": round(left_clearance, 3),
            "right_clearance_m": round(right_clearance, 3),
            "recommended_step_m": round(recommended_step, 3),
            "last_progress_m": None if self._last_progress_m is None else round(self._last_progress_m, 3),
            "regression_streak": self._regression_streak,
            "in_known_free_space": self._is_known_free(x, y),
        }

    def move_forward(self, distance_m: float, speed: float | None = None) -> str:
        """Drive forward/backward by *distance_m* meters using cmd_vel."""
        if speed is None:
            speed = self._linear_speed
        speed = min(abs(speed), 0.26)

        x0, y0, yaw0 = self.get_pose()
        goal_before = math.hypot(self._dest_x - x0, self._dest_y - y0)
        forward_clearance = self._clearance_in_direction(x0, y0, yaw0 if distance_m >= 0 else yaw0 + math.pi)
        heading_to_goal = math.atan2(self._dest_y - y0, self._dest_x - x0)
        heading_error_deg = abs(math.degrees(math.atan2(math.sin(heading_to_goal - yaw0), math.cos(heading_to_goal - yaw0))))
        requested_dist = abs(distance_m)
        target_dist = requested_dist
        direction = 1.0 if distance_m >= 0 else -1.0
        if direction > 0:
            safe_dist = self._recommended_step(
                forward_clearance,
                heading_error_deg,
                goal_before,
                self._regression_streak,
            )
            if safe_dist <= 0.05:
                return (
                    f"Heuristic blocked forward move: clearance {forward_clearance:.2f} m, "
                    f"heading error {heading_error_deg:.1f} deg."
                )
            target_dist = safe_dist
        else:
            target_dist = min(requested_dist, 0.4)

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
        if progress < -0.05:
            self._regression_streak += 1
        elif progress > 0.05:
            self._regression_streak = 0
        details = []
        # Report the controller-selected step size for this move.
        details.append(f"step {target_dist:.2f} m")
        # If a reverse request was too large, report the capped distance we actually allowed.
        if direction < 0 and target_dist + 1e-6 < requested_dist:
            details.append(f"Rejected distance: {requested_dist:.2f} m to {target_dist:.2f} m")
        # report moved away from the goal, or made progress toward the goal, if it's meaningful compared to odometry noise.
        if progress < -0.05:
            details.append(f"warning: goal distance increased by {abs(progress):.2f} m")
        else:
            details.append(f"progress {progress:.2f} m")
        # If recent moves have made things worse, include the current regression streak.
        if self._regression_streak > 0:
            details.append(f"regression streak {self._regression_streak}")
        suffix = f" ({'; '.join(details)})" if details else ""
        return f"Moved {traveled:.2f} m. Now at ({cx:.2f}, {cy:.2f}).{suffix}"

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
        forward_clearance: float,
        heading_error_deg: float,
        distance_to_goal: float,
        regression_streak: int,
    ) -> float:
        if heading_error_deg > 35.0:
            return 0.0

        # how much free space should i leave as a buffer
        usable_clearance = max(0.0, forward_clearance - 0.10)
        if usable_clearance <= 0.05:
            return 0.0

        # small usable distance
        if usable_clearance < 0.15:
            step_m = 0.08
        elif distance_to_goal <= 0.75:
            step_m = min(distance_to_goal, usable_clearance, 0.20)
        elif distance_to_goal <= 2.0:
            step_m = min(distance_to_goal, usable_clearance, 0.35)
        else:
            step_m = min(usable_clearance, 0.50)

        if regression_streak >= 2:
            step_m = min(step_m, 0.15)

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
    # Agent loop (background thread)
    # ------------------------------------------------------------------

    def _run_agent(self) -> None:
        """Wait for odom alignment, build the agent, and run the navigation loop."""
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

        for step in range(1, max_steps + 1):
            if not rclpy.ok():
                break

            self._publish_status(f"Agent step {step}/{max_steps}")

            try:
                summary = agent.step()
                self._publish_status(f"  -> {summary}")
                if not logged_run_dir and agent.run_dir:
                    self._publish_status(f"Debug images saved to: {agent.run_dir}")
                    logged_run_dir = True
            except Exception as exc:
                self._publish_status(f"Agent error on step {step}: {exc}")
                self.get_logger().error(f"Agent step {step} failed: {exc}")
                break

            x, y, _ = self.get_pose()
            dist = math.hypot(self._dest_x - x, self._dest_y - y)
            if dist < goal_tol:
                self._publish_status(
                    f"GOAL REACHED after {step} steps! "
                    f"Distance: {dist:.2f} m (tolerance: {goal_tol:.2f} m)"
                )
                return

            if agent.goal_reached_in_last_step:
                self._publish_status(
                    f"Agent reports goal reached after {step} steps."
                )
                return

        x, y, _ = self.get_pose()
        dist = math.hypot(self._dest_x - x, self._dest_y - y)
        self._publish_status(
            f"Agent exhausted {max_steps} steps. "
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
