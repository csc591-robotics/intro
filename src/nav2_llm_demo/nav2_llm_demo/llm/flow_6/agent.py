"""Flow_6 agent: Nav2 ``NavigateToPose`` wrapper.

Public surface mirrors the LLM flows so the ROS node never needs
per-flow conditionals: ``initialize / step / terminated /
goal_reached_in_last_step / run_dir``.

Cycle layout (driven by the existing ROS while-loop in
``llm_agent_node._run_agent``, one call per iteration):

    1. ``initialize`` is called once with source/destination map-frame
       coords. We use ``get_controller()`` to grab the live ROS node,
       create an ``ActionClient(node, NavigateToPose,
       '/navigate_to_pose')``, wait for the BT navigator to come up,
       and send a single goal at the destination pose.
    2. Each ``step()`` blocks ~1s, returns a one-line summary of the
       latest Nav2 feedback (``distance_remaining`` / recoveries), and
       publishes nothing to ``/cmd_vel`` directly — Nav2's controller
       owns the wheels.
    3. The action goal's result callback flips
       ``goal_reached_in_last_step`` (on ``STATUS_SUCCEEDED``) or
       ``terminated`` (on aborted / cancelled), and the surrounding
       loop in ``llm_agent_node`` publishes the matching terminal
       string on ``/navigation_status`` so the recorder writes
       ``done.flag``.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from rclpy.node import Node

from ..controller import get_controller, make_run_dir


# Roughly the LLM flows' goal radius. The outer loop in
# ``llm_agent_node`` *also* checks geometric distance < goal_tol after
# every step, so this value mainly matters for the action's own
# precision tolerance (Nav2 sets it via params).
GOAL_REACHED_THRESHOLD_M = 0.3

# How long ``step()`` blocks before returning a status line. Keeps the
# outer ROS loop from tight-spinning while still letting the orchestrator
# notice ``terminated`` / ``goal_reached_in_last_step`` quickly. Each
# ``step()`` returns immediately once Nav2 reports a terminal status,
# so this is just an upper bound per cycle. With ``max_steps=50`` (the
# default in ``experiments.yaml``) this gives Nav2 up to ~250s of
# wall-clock, which is plenty for the warehouse routes (~15 m at
# ~0.25 m/s ≈ 60 s).
STEP_PERIOD_SEC = 5.0

# How long we'll wait for ``/navigate_to_pose`` to appear before giving
# up. The Nav2 BT navigator typically takes a few seconds after launch
# to finish lifecycle activation.
ACTION_SERVER_WAIT_SEC = 60.0

# Even after ``wait_for_server`` returns, the rest of the Nav2 lifecycle
# (planner, controller, costmaps) often needs a few extra seconds —
# especially when the robot is still being spawned by Gazebo. Sleeping
# briefly here dramatically reduces "Nav2 rejected the goal" races.
NAV2_SETTLE_SEC = 6.0

# When a goal is rejected (typically because a costmap is still
# configuring / activating), retry rather than terminating the run.
GOAL_RETRY_LIMIT = 6
GOAL_RETRY_DELAY_SEC = 3.0


def _quat_from_yaw(yaw: float) -> tuple[float, float, float, float]:
    import math
    half = 0.5 * yaw
    return (0.0, 0.0, math.sin(half), math.cos(half))


class Flow6Agent:
    """Pure Nav2 ``NavigateToPose`` runner."""

    def __init__(self) -> None:
        self._node: Node | None = None
        self._action_client: ActionClient | None = None
        self._goal_handle = None
        self._initialized = False

        self._lock = threading.Lock()
        self._latest_distance_remaining: float | None = None
        self._latest_recoveries: int | None = None
        self._latest_eta_sec: float | None = None
        self._cached_summary: str = "Flow_6 idle"
        self._final_status_code: int | None = None
        self._retries_remaining: int = GOAL_RETRY_LIMIT
        # Set when the action goal reaches a terminal state — lets
        # ``step()`` return immediately on success/failure instead of
        # eating up the rest of its sleep budget.
        self._terminal_event = threading.Event()

        self.terminated: bool = False
        self.goal_reached_in_last_step: bool = False

        self.source_x: float = 0.0
        self.source_y: float = 0.0
        self.dest_x: float = 0.0
        self.dest_y: float = 0.0

        self.run_dir: Path = make_run_dir(flow="6")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(
        self,
        source_x: float,
        source_y: float,
        dest_x: float,
        dest_y: float,
    ) -> None:
        """Spin up the action client and dispatch the single Nav2 goal."""
        self.source_x = float(source_x)
        self.source_y = float(source_y)
        self.dest_x = float(dest_x)
        self.dest_y = float(dest_y)

        ctrl = get_controller()
        if not isinstance(ctrl, Node):
            raise RuntimeError(
                "Flow_6 requires the controller to be a rclpy.node.Node "
                "(it is, in production). Got: " + repr(type(ctrl))
            )
        self._node = ctrl

        self._action_client = ActionClient(
            self._node, NavigateToPose, "/navigate_to_pose",
        )

        ok = self._action_client.wait_for_server(
            timeout_sec=ACTION_SERVER_WAIT_SEC,
        )
        if not ok:
            self.terminated = True
            self._cached_summary = (
                "Agent terminated: Nav2 /navigate_to_pose action server "
                "did not appear within "
                f"{ACTION_SERVER_WAIT_SEC:.0f}s."
            )
            return

        # Give the rest of the Nav2 stack a chance to finish activating
        # before we punch in the goal. Without this, the BT navigator's
        # action server is up but planner/controller/costmaps may still
        # be configuring, and we get an immediate "goal rejected".
        time.sleep(NAV2_SETTLE_SEC)

        self._send_goal()

        self._initialized = True
        self._cached_summary = (
            f"Nav2 goal sent to ({self.dest_x:.2f}, {self.dest_y:.2f})."
        )

    def _send_goal(self) -> None:
        send_future = self._action_client.send_goal_async(
            self._build_goal(),
            feedback_callback=self._on_feedback,
        )
        send_future.add_done_callback(self._on_goal_response)

    def step(self) -> str:
        """Block up to one period and return a one-line status summary.

        Returns early as soon as Nav2 reports a terminal status so the
        outer loop can publish ``GOAL REACHED`` / ``Agent terminated``
        without waiting for the rest of the sleep budget.
        """
        self._terminal_event.wait(timeout=STEP_PERIOD_SEC)

        with self._lock:
            if self._final_status_code is not None:
                return self._final_summary_locked()

            if self._latest_distance_remaining is not None:
                dist = self._latest_distance_remaining
                rec = self._latest_recoveries or 0
                eta_part = ""
                if self._latest_eta_sec is not None:
                    eta_part = (
                        f", eta={self._latest_eta_sec:.1f}s"
                    )
                self._cached_summary = (
                    f"Nav2: distance_remaining={dist:.2f} m, "
                    f"recoveries={rec}{eta_part}"
                )
            return self._cached_summary

    # ------------------------------------------------------------------
    # Action client glue (callbacks run on rclpy executor threads)
    # ------------------------------------------------------------------

    def _build_goal(self) -> NavigateToPose.Goal:
        goal = NavigateToPose.Goal()
        goal.pose = PoseStamped()
        goal.pose.header.frame_id = "map"
        goal.pose.header.stamp = self._node.get_clock().now().to_msg()
        goal.pose.pose.position.x = self.dest_x
        goal.pose.pose.position.y = self.dest_y
        goal.pose.pose.position.z = 0.0
        qx, qy, qz, qw = _quat_from_yaw(0.0)
        goal.pose.pose.orientation.x = qx
        goal.pose.pose.orientation.y = qy
        goal.pose.pose.orientation.z = qz
        goal.pose.pose.orientation.w = qw
        return goal

    def _on_feedback(self, feedback_msg) -> None:
        fb = feedback_msg.feedback
        with self._lock:
            try:
                self._latest_distance_remaining = float(fb.distance_remaining)
            except (AttributeError, TypeError, ValueError):
                self._latest_distance_remaining = None
            try:
                self._latest_recoveries = int(fb.number_of_recoveries)
            except (AttributeError, TypeError, ValueError):
                self._latest_recoveries = None
            try:
                eta = fb.estimated_time_remaining
                self._latest_eta_sec = float(eta.sec) + float(
                    eta.nanosec
                ) * 1e-9
            except (AttributeError, TypeError, ValueError):
                self._latest_eta_sec = None

    def _on_goal_response(self, future) -> None:
        try:
            goal_handle = future.result()
        except Exception as exc:
            with self._lock:
                self._final_status_code = -1
                self._cached_summary = (
                    f"Agent terminated: Nav2 send_goal failed: {exc}"
                )
            self.terminated = True
            self._terminal_event.set()
            return

        if not goal_handle.accepted:
            # Most "rejected" cases are transient: the BT navigator was
            # up but a costmap or the controller wasn't yet active. Retry
            # a handful of times before giving up so a slow Nav2
            # cold-start doesn't kill the whole run.
            with self._lock:
                if self._retries_remaining > 0:
                    self._retries_remaining -= 1
                    self._cached_summary = (
                        "Nav2 rejected the goal (likely lifecycle still "
                        "activating); retrying in "
                        f"{GOAL_RETRY_DELAY_SEC:.1f}s "
                        f"({self._retries_remaining} retries left)."
                    )
                    threading.Timer(
                        GOAL_RETRY_DELAY_SEC, self._send_goal,
                    ).start()
                    return
                self._final_status_code = GoalStatus.STATUS_ABORTED
                self._cached_summary = (
                    "Agent terminated: Nav2 rejected the goal "
                    f"after {GOAL_RETRY_LIMIT} retries."
                )
            self.terminated = True
            self._terminal_event.set()
            return

        self._goal_handle = goal_handle
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._on_result)

    def _on_result(self, future) -> None:
        try:
            wrapped = future.result()
        except Exception as exc:
            with self._lock:
                self._final_status_code = -1
                self._cached_summary = (
                    f"Agent terminated: Nav2 get_result failed: {exc}"
                )
            self.terminated = True
            self._terminal_event.set()
            return

        status = wrapped.status
        with self._lock:
            self._final_status_code = status
            if status == GoalStatus.STATUS_SUCCEEDED:
                self._cached_summary = (
                    "GOAL REACHED via Nav2 NavigateToPose action."
                )
            elif status == GoalStatus.STATUS_CANCELED:
                self._cached_summary = (
                    "Agent exhausted: Nav2 goal canceled."
                )
            elif status == GoalStatus.STATUS_ABORTED:
                self._cached_summary = (
                    "Agent terminated: Nav2 aborted the goal "
                    "(planner or controller failure)."
                )
            else:
                self._cached_summary = (
                    f"Agent terminated: Nav2 ended with status {status}."
                )

        if status == GoalStatus.STATUS_SUCCEEDED:
            self.goal_reached_in_last_step = True
        else:
            self.terminated = True
        self._terminal_event.set()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _final_summary_locked(self) -> str:
        return self._cached_summary


def build_agent(*_args: Any, **_kwargs: Any) -> Flow6Agent:
    """Factory matching the LLM flows' ``build_agent()`` shape."""
    return Flow6Agent()
