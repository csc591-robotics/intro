"""ROS 2 node that runs a vision-based LLM agent to navigate a robot.

The node:
- Subscribes to /odom for robot position tracking
- Subscribes to /scan to expose live LiDAR data to flow_3
- Publishes to /cmd_vel for direct robot control
- Uses TF2 to get accurate map->base_link transforms
- Runs a VisionNavigationAgent that sees an annotated map and calls tools
"""

import json
import math
import os
import threading
import time
from pathlib import Path
from typing import Any

from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
import tf2_ros

# Registers geometry_msgs / PoseStamped converters for Buffer.transform().
import tf2_geometry_msgs.tf2_geometry_msgs  # noqa: F401

from .llm import build_agent, set_controller


def _yaw_from_quat(qx: float, qy: float, qz: float, qw: float) -> float:
    siny = 2.0 * (qw * qz + qx * qy)
    cosy = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny, cosy)


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
        # Strict goal radius: ~robot footprint (override per-launch if needed).
        self.declare_parameter("goal_tolerance_m", 0.3)
        # 0 = unlimited; loop runs until goal reached or rclpy shuts down.
        self.declare_parameter("max_agent_steps", 0)
        self.declare_parameter("linear_speed", 0.15)
        self.declare_parameter("angular_speed", 0.5)
        self.declare_parameter("status_topic", "/navigation_status")
        self.declare_parameter("move_timeout_sec", 30.0)

        self._source_x = self._float("source_x")
        self._source_y = self._float("source_y")
        self._dest_x = self._float("dest_x")
        self._dest_y = self._float("dest_y")
        self._map_yaml = self._str("map_yaml_path")
        self._linear_speed = self._float("linear_speed")
        self._angular_speed = self._float("angular_speed")
        self._move_timeout = self._float("move_timeout_sec")

        self._pose_lock = threading.Lock()
        self._odom_x = self._source_x
        self._odom_y = self._source_y
        self._odom_yaw = self._float("source_yaw")
        self._pose_ready = False
        self._last_odom_pose: PoseStamped | None = None
        self._logged_tf_fallback = False

        self._scan_lock = threading.Lock()
        self._latest_scan: dict[str, Any] | None = None

        self._cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        status_topic = self._str("status_topic")
        self._status_pub = self.create_publisher(String, status_topic, 10)

        self._odom_sub = self.create_subscription(
            Odometry, "/odom", self._odom_cb, 10,
        )
        self._scan_sub = self.create_subscription(
            LaserScan, "/scan", self._scan_cb, 10,
        )

        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        # Optional per-action JSONL log. The nav2_llm_experiments orchestrator
        # sets this to <experiment_dir>/actions.jsonl so every move_forward /
        # rotate call gets a structured record (start/end ts, requested vs
        # achieved, timed_out). When the env var is unset the writes are
        # skipped, keeping the standalone demo behavior unchanged.
        self._actions_log_path: Path | None = None
        actions_log_env = os.environ.get("LLM_ACTIONS_LOG", "").strip()
        if actions_log_env:
            self._actions_log_path = Path(actions_log_env).expanduser()
            try:
                self._actions_log_path.parent.mkdir(parents=True, exist_ok=True)
            except OSError:
                self._actions_log_path = None
        self._actions_log_lock = threading.Lock()

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
        """Return (x, y, yaw) in the map frame.

        TurtleBot3 Gazebo publishes ``odom -> base_footprint``.  We prefer that
        chain over ``map -> base_link`` so the first lookup succeeds reliably.

        If direct map->base lookups fail, we transform the latest ``/odom``
        pose from ``odom`` into ``map``.  Raw odom (x, y) must **not** be used
        as map coordinates — that mis-centers map crops for the vision LLM.
        """
        timeout = Duration(seconds=2.0)
        for child in ("base_footprint", "base_link"):
            try:
                t = self._tf_buffer.lookup_transform(
                    "map", child, rclpy.time.Time(), timeout=timeout,
                )
                x = t.transform.translation.x
                y = t.transform.translation.y
                yaw = _yaw_from_quat(
                    t.transform.rotation.x,
                    t.transform.rotation.y,
                    t.transform.rotation.z,
                    t.transform.rotation.w,
                )
                with self._pose_lock:
                    self._odom_x = x
                    self._odom_y = y
                    self._odom_yaw = yaw
                return x, y, yaw
            except (
                tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException,
            ):
                timeout = Duration(seconds=0.5)

        with self._pose_lock:
            ps = self._last_odom_pose
        if ps is not None:
            try:
                pm = self._tf_buffer.transform(
                    ps, "map", timeout=Duration(seconds=0.5),
                )
                x = pm.pose.position.x
                y = pm.pose.position.y
                yaw = _yaw_from_quat(
                    pm.pose.orientation.x,
                    pm.pose.orientation.y,
                    pm.pose.orientation.z,
                    pm.pose.orientation.w,
                )
                if not self._logged_tf_fallback:
                    self.get_logger().warn(
                        "map->base_* TF lookup failed; using odom pose transformed "
                        "into map for navigation / map crops."
                    )
                    self._logged_tf_fallback = True
                with self._pose_lock:
                    self._odom_x = x
                    self._odom_y = y
                    self._odom_yaw = yaw
                return x, y, yaw
            except (
                tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException,
            ):
                pass

        with self._pose_lock:
            return self._odom_x, self._odom_y, self._odom_yaw

    def move_forward(self, distance_m: float, speed: float | None = None) -> str:
        """Drive forward/backward by *distance_m* meters using cmd_vel."""
        if speed is None:
            speed = self._linear_speed
        speed = min(abs(speed), 0.26)

        x0, y0, yaw0 = self.get_pose()
        target_dist = abs(distance_m)
        direction = 1.0 if distance_m >= 0 else -1.0

        twist = Twist()
        twist.linear.x = direction * speed

        t_start_wall = time.time()
        t0 = time.monotonic()
        rate = self.create_rate(20)
        traveled = 0.0
        timed_out = False

        while traveled < target_dist:
            if time.monotonic() - t0 > self._move_timeout:
                self._stop()
                cx, cy, cyaw = self.get_pose()
                self._log_action(
                    "move_forward",
                    {"distance_m": float(distance_m), "speed": float(speed)},
                    t_start_wall,
                    (x0, y0, yaw0),
                    (cx, cy, cyaw),
                    achieved={"distance_m": float(traveled)},
                    timed_out=True,
                )
                return f"Timed out after moving {traveled:.2f} m of requested {distance_m:.2f} m."

            self._cmd_pub.publish(twist)
            rate.sleep()

            cx, cy, _ = self.get_pose()
            traveled = math.hypot(cx - x0, cy - y0)

        self._stop()
        cx, cy, cyaw = self.get_pose()
        self._log_action(
            "move_forward",
            {"distance_m": float(distance_m), "speed": float(speed)},
            t_start_wall,
            (x0, y0, yaw0),
            (cx, cy, cyaw),
            achieved={"distance_m": float(traveled)},
            timed_out=timed_out,
        )
        return f"Moved {traveled:.2f} m. Now at ({cx:.2f}, {cy:.2f})."

    def rotate(self, angle_deg: float, speed: float | None = None) -> str:
        """Rotate in place by *angle_deg* degrees."""
        if speed is None:
            speed = self._angular_speed
        speed = min(abs(speed), 1.0)

        angle_rad = math.radians(angle_deg)
        target_rad = abs(angle_rad)
        direction = 1.0 if angle_rad >= 0 else -1.0

        x0, y0, yaw0 = self.get_pose()

        twist = Twist()
        twist.angular.z = direction * speed

        t_start_wall = time.time()
        t0 = time.monotonic()
        rate = self.create_rate(20)
        rotated = 0.0
        prev_yaw = yaw0
        timed_out = False

        while rotated < target_rad:
            if time.monotonic() - t0 > self._move_timeout:
                self._stop()
                cx, cy, cyaw = self.get_pose()
                self._log_action(
                    "rotate",
                    {"angle_deg": float(angle_deg), "speed": float(speed)},
                    t_start_wall,
                    (x0, y0, yaw0),
                    (cx, cy, cyaw),
                    achieved={"angle_deg": math.degrees(rotated)},
                    timed_out=True,
                )
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
        cx, cy, final_yaw = self.get_pose()
        self._log_action(
            "rotate",
            {"angle_deg": float(angle_deg), "speed": float(speed)},
            t_start_wall,
            (x0, y0, yaw0),
            (cx, cy, final_yaw),
            achieved={"angle_deg": math.degrees(rotated)},
            timed_out=timed_out,
        )
        return f"Rotated {math.degrees(rotated):.1f} deg. Heading now {math.degrees(final_yaw):.1f} deg."

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _stop(self) -> None:
        self._cmd_pub.publish(Twist())

    def _log_action(
        self,
        name: str,
        args: dict[str, Any],
        t_start_wall: float,
        pose_before: tuple[float, float, float],
        pose_after: tuple[float, float, float],
        achieved: dict[str, float],
        timed_out: bool,
    ) -> None:
        """Append one JSONL record to ``LLM_ACTIONS_LOG`` if it is set.

        The experiments orchestrator points ``LLM_ACTIONS_LOG`` at
        ``<experiment_dir>/actions.jsonl``. Demo runs leave it unset and
        this method becomes a no-op.
        """
        if self._actions_log_path is None:
            return
        record = {
            "name": name,
            "args": args,
            "t_start": t_start_wall,
            "t_end": time.time(),
            "duration_sec": time.time() - t_start_wall,
            "pose_before": {
                "x": pose_before[0], "y": pose_before[1], "yaw": pose_before[2],
            },
            "pose_after": {
                "x": pose_after[0], "y": pose_after[1], "yaw": pose_after[2],
            },
            "achieved": achieved,
            "timed_out": bool(timed_out),
        }
        line = json.dumps(record, ensure_ascii=False) + "\n"
        try:
            with self._actions_log_lock:
                with self._actions_log_path.open("a", encoding="utf-8") as fh:
                    fh.write(line)
        except OSError:
            pass

    def _odom_cb(self, msg: Odometry) -> None:
        pos = msg.pose.pose.position
        ori = msg.pose.pose.orientation
        yaw = _yaw_from_quat(ori.x, ori.y, ori.z, ori.w)
        ps = PoseStamped()
        ps.header = msg.header
        ps.pose = msg.pose.pose
        with self._pose_lock:
            self._odom_x = pos.x
            self._odom_y = pos.y
            self._odom_yaw = yaw
            self._last_odom_pose = ps
            self._pose_ready = True

    def _scan_cb(self, msg: LaserScan) -> None:
        """Snapshot the latest LaserScan as a JSON-friendly dict so flow tools
        can consume it without importing sensor_msgs."""
        snapshot: dict[str, Any] = {
            "ranges": list(msg.ranges),
            "angle_min": float(msg.angle_min),
            "angle_increment": float(msg.angle_increment),
            "range_min": float(msg.range_min),
            "range_max": float(msg.range_max),
            "frame_id": msg.header.frame_id,
            "stamp_sec": msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
        }
        with self._scan_lock:
            self._latest_scan = snapshot

    def get_latest_scan(self) -> dict[str, Any] | None:
        """RobotController hook: return the latest LaserScan dict, or None."""
        with self._scan_lock:
            return self._latest_scan

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
        """Wait for odom, build the agent, and run the navigation loop."""
        self.get_logger().info("Waiting for odometry data...")
        while rclpy.ok():
            with self._pose_lock:
                if self._pose_ready:
                    break
            time.sleep(0.2)

        self._publish_status(
            f"Odometry received. Starting navigation from "
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

        # max_agent_steps == 0 -> unlimited (loop until goal or shutdown).
        # The nav2_llm_experiments orchestrator overrides this via the env
        # var LLM_AGENT_MAX_STEPS (default 50 in that orchestrator) without
        # needing to thread a launch arg through. The launch-arg parameter
        # still wins when explicitly set; the env var only fills in the
        # default ``0`` (unlimited) so demo behavior is unchanged.
        max_steps = self._int("max_agent_steps")
        if max_steps <= 0:
            env_val = os.environ.get("LLM_AGENT_MAX_STEPS", "").strip()
            if env_val:
                try:
                    max_steps = max(0, int(env_val))
                except ValueError:
                    self.get_logger().warn(
                        f"Ignoring non-integer LLM_AGENT_MAX_STEPS={env_val!r}"
                    )
        goal_tol = self._float("goal_tolerance_m")
        logged_run_dir = False

        step = 0
        while rclpy.ok():
            step += 1
            if max_steps > 0 and step > max_steps:
                break

            cap = str(max_steps) if max_steps > 0 else "unlimited"
            self._publish_status(f"Agent step {step}/{cap}")

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

            # ReAct flows finish a whole episode in a single step() (the
            # graph runs LLM<->tool internally until it stops). After that
            # the agent caches its final summary and returns it instantly,
            # so without this bail the ROS while-loop tight-spins forever
            # on the cached string (e.g. when the LLM hits a 429).
            if getattr(agent, "terminated", False):
                self._publish_status(
                    f"Agent terminated after {step} steps. "
                    f"Final: {summary}"
                )
                return

        x, y, _ = self.get_pose()
        dist = math.hypot(self._dest_x - x, self._dest_y - y)
        if max_steps > 0:
            self._publish_status(
                f"Agent exhausted {max_steps} steps. "
                f"Distance to goal: {dist:.2f} m."
            )
        else:
            self._publish_status(
                f"Agent loop ended after {step} steps. "
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
