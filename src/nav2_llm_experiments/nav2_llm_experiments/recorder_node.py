"""Per-experiment data recorder for ``nav2_llm_experiments``.

Subscribes to the topics produced by ``nav2_llm_demo`` during a single
experiment run and writes append-only JSONL streams into the experiment's
output directory. Also watches ``/navigation_status`` for terminal
substrings and touches ``done.flag`` so the orchestrator knows the agent
is finished and it is safe to SIGINT the launch tree.

Usage (the orchestrator spawns this directly; you should not need to run
it by hand):

    ros2 run nav2_llm_experiments recorder_node \\
        --ros-args -p output_dir:=/path/to/experiment_data_folder/...

Append-only JSONL is used everywhere so a crash mid-experiment still
leaves a valid prefix-decoded file (no half-written JSON object trees).
"""

from __future__ import annotations

import json
import math
import os
import threading
import time
from pathlib import Path
from typing import Any

import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
import tf2_ros


# Substrings published to ``/navigation_status`` by ``llm_agent_node``
# that signal the agent has finished (success or terminal failure).
TERMINAL_STATUS_SUBSTRINGS = (
    "GOAL REACHED",
    "Agent reports goal reached",
    "Agent terminated",
    "Agent exhausted",
    "Agent loop ended",
    "Failed to build LLM agent",
    "A* planner failed",
    "A* found no path",
)


def _yaw_from_quat(qx: float, qy: float, qz: float, qw: float) -> float:
    siny = 2.0 * (qw * qz + qx * qy)
    cosy = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny, cosy)


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + "Z"


class _JsonlWriter:
    """Thread-safe append-only JSONL writer.

    Each record is one line of JSON terminated by ``\\n``. We open the file
    in append mode per record (slow but bulletproof for crashes) when
    ``flush_each=True``; otherwise the file handle is held open and
    ``flush()``'d after each record to keep tail-following sane.
    """

    def __init__(self, path: Path, flush_each: bool = False) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._flush_each = flush_each
        self._fh = None
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if not flush_each:
            self._fh = self._path.open("a", encoding="utf-8")

    def write(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, ensure_ascii=False, default=str) + "\n"
        try:
            with self._lock:
                if self._flush_each or self._fh is None:
                    with self._path.open("a", encoding="utf-8") as fh:
                        fh.write(line)
                else:
                    self._fh.write(line)
                    self._fh.flush()
        except OSError:
            pass

    def close(self) -> None:
        with self._lock:
            if self._fh is not None:
                try:
                    self._fh.flush()
                    self._fh.close()
                except OSError:
                    pass
                self._fh = None


class RecorderNode(Node):
    """Subscribes to /odom, /scan, /cmd_vel, /navigation_status and TF.

    Writes:

    - ``pose_stream.jsonl``      — odom-frame pose + twist (raw /odom)
    - ``pose_map_stream.jsonl``  — map -> base_link TF lookup at 10 Hz
    - ``scan_stream.jsonl``      — LaserScan summary (8 angular sectors)
    - ``cmd_vel_stream.jsonl``   — every Twist published to /cmd_vel
    - ``status.log``             — every /navigation_status string
    - ``done.flag``              — touched the moment a terminal status
                                   string is seen on /navigation_status
    """

    def __init__(self) -> None:
        super().__init__("nav2_llm_experiments_recorder")

        self.declare_parameter("output_dir", "")
        self.declare_parameter("scan_sectors", 8)
        self.declare_parameter("tf_sample_hz", 10.0)
        self.declare_parameter("use_sim_time_param", True)

        out_str = self.get_parameter("output_dir") \
            .get_parameter_value().string_value.strip()
        if not out_str:
            raise RuntimeError(
                "RecorderNode requires the output_dir parameter."
            )

        self._out_dir = Path(out_str).expanduser().resolve()
        self._out_dir.mkdir(parents=True, exist_ok=True)
        self._done_flag = self._out_dir / "done.flag"
        self._terminal_status: str | None = None
        self._terminal_status_lock = threading.Lock()

        self._pose_writer = _JsonlWriter(self._out_dir / "pose_stream.jsonl")
        self._pose_map_writer = _JsonlWriter(
            self._out_dir / "pose_map_stream.jsonl"
        )
        self._scan_writer = _JsonlWriter(self._out_dir / "scan_stream.jsonl")
        self._cmd_writer = _JsonlWriter(self._out_dir / "cmd_vel_stream.jsonl")
        self._status_log_path = self._out_dir / "status.log"
        self._status_log_lock = threading.Lock()

        self._scan_sectors = max(
            1,
            int(self.get_parameter("scan_sectors")
                .get_parameter_value().integer_value or 8),
        )
        tf_hz = float(
            self.get_parameter("tf_sample_hz")
                .get_parameter_value().double_value or 10.0
        )
        tf_period = 1.0 / max(0.5, tf_hz)

        # /navigation_status is published with default QoS by the agent
        # (depth=10, reliable, volatile).  /odom and /scan in the
        # TurtleBot3 stack also use sensor-data style QoS — best-effort
        # is safer for /scan to avoid dropping when bandwidth is high.
        scan_qos = QoSProfile(
            depth=10,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
        )

        self.create_subscription(Odometry, "/odom", self._odom_cb, 50)
        self.create_subscription(LaserScan, "/scan", self._scan_cb, scan_qos)
        self.create_subscription(Twist, "/cmd_vel", self._cmd_cb, 50)
        self.create_subscription(
            String, "/navigation_status", self._status_cb, 10,
        )

        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)
        self.create_timer(tf_period, self._tf_tick)

        self._t_start_wall = time.time()
        try:
            (self._out_dir / "recorder_started.txt").write_text(
                _now_iso() + "\n", encoding="utf-8",
            )
        except OSError:
            pass
        self.get_logger().info(
            f"Recorder writing to {self._out_dir} "
            f"(scan_sectors={self._scan_sectors}, tf_hz={tf_hz})"
        )

    # ------------------------------------------------------------------
    # Subscriptions
    # ------------------------------------------------------------------

    def _odom_cb(self, msg: Odometry) -> None:
        pos = msg.pose.pose.position
        ori = msg.pose.pose.orientation
        twist = msg.twist.twist
        self._pose_writer.write({
            "t_wall": time.time(),
            "stamp_sec": msg.header.stamp.sec
                + msg.header.stamp.nanosec * 1e-9,
            "frame_id": msg.header.frame_id,
            "x": float(pos.x),
            "y": float(pos.y),
            "z": float(pos.z),
            "qx": float(ori.x),
            "qy": float(ori.y),
            "qz": float(ori.z),
            "qw": float(ori.w),
            "yaw": _yaw_from_quat(ori.x, ori.y, ori.z, ori.w),
            "lin_x": float(twist.linear.x),
            "lin_y": float(twist.linear.y),
            "ang_z": float(twist.angular.z),
        })

    def _scan_cb(self, msg: LaserScan) -> None:
        ranges = list(msg.ranges)
        n = len(ranges)
        if n == 0:
            return

        rmin = float(msg.range_min)
        rmax = float(msg.range_max)

        valid = [r for r in ranges
                 if math.isfinite(r) and rmin <= r <= rmax]
        overall_min = min(valid) if valid else None

        sectors: list[dict[str, Any]] = []
        sector_size = max(1, n // self._scan_sectors)
        for i in range(self._scan_sectors):
            s = i * sector_size
            e = (i + 1) * sector_size if i < self._scan_sectors - 1 else n
            sector_vals = [
                r for r in ranges[s:e]
                if math.isfinite(r) and rmin <= r <= rmax
            ]
            if sector_vals:
                sectors.append({
                    "i": i,
                    "min": min(sector_vals),
                    "max": max(sector_vals),
                    "mean": sum(sector_vals) / len(sector_vals),
                    "n": len(sector_vals),
                })
            else:
                sectors.append({"i": i, "n": 0})

        self._scan_writer.write({
            "t_wall": time.time(),
            "stamp_sec": msg.header.stamp.sec
                + msg.header.stamp.nanosec * 1e-9,
            "frame_id": msg.header.frame_id,
            "n_rays": n,
            "angle_min": float(msg.angle_min),
            "angle_increment": float(msg.angle_increment),
            "range_min": rmin,
            "range_max": rmax,
            "overall_min": overall_min,
            "sectors": sectors,
        })

    def _cmd_cb(self, msg: Twist) -> None:
        self._cmd_writer.write({
            "t_wall": time.time(),
            "lin_x": float(msg.linear.x),
            "lin_y": float(msg.linear.y),
            "lin_z": float(msg.linear.z),
            "ang_x": float(msg.angular.x),
            "ang_y": float(msg.angular.y),
            "ang_z": float(msg.angular.z),
        })

    def _status_cb(self, msg: String) -> None:
        text = msg.data
        line = f"[{_now_iso()}] {text}\n"
        try:
            with self._status_log_lock:
                with self._status_log_path.open("a", encoding="utf-8") as fh:
                    fh.write(line)
        except OSError:
            pass

        for needle in TERMINAL_STATUS_SUBSTRINGS:
            if needle in text:
                with self._terminal_status_lock:
                    if self._terminal_status is None:
                        self._terminal_status = text
                self._touch_done(text)
                return

    def _tf_tick(self) -> None:
        try:
            t = self._tf_buffer.lookup_transform(
                "map", "base_footprint",
                rclpy.time.Time(),
                timeout=Duration(seconds=0.0),
            )
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ):
            try:
                t = self._tf_buffer.lookup_transform(
                    "map", "base_link",
                    rclpy.time.Time(),
                    timeout=Duration(seconds=0.0),
                )
            except (
                tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException,
            ):
                return
        tr = t.transform.translation
        rot = t.transform.rotation
        self._pose_map_writer.write({
            "t_wall": time.time(),
            "stamp_sec": t.header.stamp.sec
                + t.header.stamp.nanosec * 1e-9,
            "frame_id": t.header.frame_id,
            "child_frame_id": t.child_frame_id,
            "x": float(tr.x),
            "y": float(tr.y),
            "z": float(tr.z),
            "qx": float(rot.x),
            "qy": float(rot.y),
            "qz": float(rot.z),
            "qw": float(rot.w),
            "yaw": _yaw_from_quat(rot.x, rot.y, rot.z, rot.w),
        })

    def _touch_done(self, reason: str) -> None:
        try:
            payload = json.dumps({
                "reason": reason,
                "t_wall": time.time(),
                "iso": _now_iso(),
            })
            self._done_flag.write_text(payload + "\n", encoding="utf-8")
        except OSError:
            pass

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        self._pose_writer.close()
        self._pose_map_writer.close()
        self._scan_writer.close()
        self._cmd_writer.close()


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node: RecorderNode | None = None
    try:
        node = RecorderNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            try:
                node.shutdown()
            finally:
                node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
