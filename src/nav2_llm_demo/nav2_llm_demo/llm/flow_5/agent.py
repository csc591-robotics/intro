"""Flow_5 agent: deterministic A* planner + LLM follower.

Public surface mirrors flow_4 so the ROS node never needs per-flow
conditionals: ``initialize / step / terminated / goal_reached_in_last_step
/ run_dir``.

Cycle layout (driven by the existing ROS while-loop, one call per
iteration):

    1. Auto-advance ``self._target_idx`` while the robot is within
       0.5 m of ``planned_path[target_idx]``.
    2. Build the situation HumanMessage (map image + suggestion text).
    3. Call the follower LLM with ``tools = [move_forward, rotate]``
       and ``tool_choice="any"``.
    4. Dispatch the chosen tool against the controller.
    5. Check distance to the FINAL destination. If < 1.0 m, terminate
       with ``GOAL REACHED``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from langchain.chat_models import init_chat_model
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from ..controller import get_controller, make_run_dir, resolve_llm_config
from ..flow_2.logging import PerCallLogger
from ..flow_3.tools import move_forward, rotate
from ..message_utils import compact_history, prune_old_images
from .path_planner import plan_astar_path
from .prompt import FOLLOWER_SYSTEM_PROMPT
from .situation import build_situation_message


GOAL_REACHED_THRESHOLD_M = 1.0
WAYPOINT_ADVANCE_THRESHOLD_M = 0.5

_DECIDE_TOOLS = [move_forward, rotate]


class Flow5Agent:
    """A* path planner + per-cycle LLM follower."""

    def __init__(
        self,
        provider: str | None = None,
        model_name: str | None = None,
    ) -> None:
        if provider is None or model_name is None:
            resolved_provider, resolved_model = resolve_llm_config("5")
            provider = provider or resolved_provider
            model_name = model_name or resolved_model

        self._provider = provider
        self._model_name = model_name

        base_llm = init_chat_model(
            model=model_name,
            model_provider=provider,
            temperature=0.0,
        )
        try:
            self._llm = base_llm.bind_tools(
                _DECIDE_TOOLS, tool_choice="any",
            )
        except TypeError:
            self._llm = base_llm.bind_tools(_DECIDE_TOOLS)

        self._messages: list[Any] = []
        self._cycle_num = 0
        self._run_dir: Path | None = None
        self._logger: PerCallLogger | None = None
        self._completed = False
        self._final_summary: str | None = None
        self._goal_marker = False

        self._planned_path: list[tuple[float, float]] = []
        self._target_idx = 0
        self._source_xy: tuple[float, float] = (0.0, 0.0)
        self._dest_xy: tuple[float, float] = (0.0, 0.0)

    # ------------------------------------------------------------------
    # Run / log directory
    # ------------------------------------------------------------------

    def _ensure_run_dir(self) -> Path:
        if self._run_dir is None:
            self._run_dir = make_run_dir(flow="5")
            self._logger = PerCallLogger(
                self._run_dir,
                provider=self._provider,
                model=self._model_name,
            )
        return self._run_dir

    # ------------------------------------------------------------------
    # Conversation setup + planning
    # ------------------------------------------------------------------

    def initialize(
        self,
        source_x: float,
        source_y: float,
        dest_x: float,
        dest_y: float,
    ) -> None:
        self._ensure_run_dir()
        self._source_xy = (source_x, source_y)
        self._dest_xy = (dest_x, dest_y)
        self._cycle_num = 0
        self._completed = False
        self._final_summary = None
        self._goal_marker = False
        self._target_idx = 0

        ctrl = get_controller()
        try:
            self._planned_path = plan_astar_path(
                map_yaml_path=ctrl.map_yaml_path,
                src_xy=(source_x, source_y),
                dst_xy=(dest_x, dest_y),
            )
        except Exception as exc:  # noqa: BLE001
            self._planned_path = []
            self._completed = True
            self._final_summary = f"A* planner failed: {type(exc).__name__}: {exc}"
            self._messages = []
            return

        if not self._planned_path:
            self._completed = True
            self._final_summary = (
                "A* found no path from "
                f"({source_x:.2f}, {source_y:.2f}) to "
                f"({dest_x:.2f}, {dest_y:.2f}). Stopping."
            )
            self._messages = []
            return

        # Preamble HumanMessage that compact_history preserves. Includes
        # the planned waypoint count so the LLM has light context.
        self._messages = [
            HumanMessage(content=(
                f"Goal: navigate from ({source_x:.2f}, {source_y:.2f}) to "
                f"({dest_x:.2f}, {dest_y:.2f}). A* has planned a path "
                f"with {len(self._planned_path)} waypoints (drawn as the "
                "MAGENTA line on every map image). Each turn I will give "
                "you the situation; reply with EXACTLY ONE move_forward "
                "or rotate tool call, defaulting to the SUGGESTED action."
            )),
        ]

    # ------------------------------------------------------------------
    # One follower cycle
    # ------------------------------------------------------------------

    def step(self) -> str:
        if self._completed:
            return self._final_summary or "Agent already completed."

        self._cycle_num += 1
        self._ensure_run_dir()

        try:
            # 0. ADVANCE TARGET WAYPOINT --------------------------------
            self._auto_advance_target_idx()

            ctrl = get_controller()
            x, y, _ = ctrl.get_pose()
            final_dist = math.hypot(
                self._dest_xy[0] - x, self._dest_xy[1] - y,
            )
            if final_dist < GOAL_REACHED_THRESHOLD_M:
                self._completed = True
                self._goal_marker = True
                self._final_summary = (
                    f"GOAL REACHED at cycle {self._cycle_num} "
                    f"(distance {final_dist:.2f} m)."
                )
                return self._final_summary

            # 1. GATHER ----------------------------------------------------
            situation_msg, _artifact = build_situation_message(
                planned_path=self._planned_path,
                target_idx=self._target_idx,
                map_yaml_path=ctrl.map_yaml_path,
                source_xy=self._source_xy,
                dest_xy=self._dest_xy,
            )
            self._messages.append(situation_msg)

            # 2. DECIDE ----------------------------------------------------
            compacted = compact_history(self._messages, keep_rounds=4)
            pruned = prune_old_images(compacted, keep_last=1)
            llm_input = (
                [SystemMessage(content=FOLLOWER_SYSTEM_PROMPT)] + pruned
            )

            config: dict[str, Any] = {
                "callbacks": [self._logger] if self._logger else [],
            }
            response = self._llm.invoke(llm_input, config=config)
            self._messages.append(response)

            tool_calls = getattr(response, "tool_calls", None) or []
            if not tool_calls:
                self._completed = True
                self._final_summary = (
                    "LLM returned no tool calls (despite tool_choice=any). "
                    "Stopping."
                )
                return self._final_summary

            # 3. EXECUTE ---------------------------------------------------
            executed_summaries: list[str] = []
            for tc in tool_calls:
                name = tc.get("name", "")
                args = tc.get("args", {}) or {}
                tc_id = tc.get("id", "")
                if name == "move_forward":
                    result = ctrl.move_forward(
                        float(args.get("distance_meters", 0.0))
                    )
                elif name == "rotate":
                    result = ctrl.rotate(
                        float(args.get("angle_degrees", 0.0))
                    )
                else:
                    result = (
                        f"Tool {name!r} is not allowed in flow_5. Only "
                        "move_forward and rotate are bound."
                    )
                executed_summaries.append(f"{name}({json.dumps(args)})")
                self._messages.append(
                    ToolMessage(content=result, tool_call_id=tc_id)
                )

            # 4. CHECK GOAL ------------------------------------------------
            self._auto_advance_target_idx()
            x, y, _ = ctrl.get_pose()
            final_dist = math.hypot(
                self._dest_xy[0] - x, self._dest_xy[1] - y,
            )
            if final_dist < GOAL_REACHED_THRESHOLD_M:
                self._completed = True
                self._goal_marker = True
                self._final_summary = (
                    f"GOAL REACHED after {self._cycle_num} cycles "
                    f"(distance {final_dist:.2f} m). "
                    f"Last action: {', '.join(executed_summaries)}."
                )
                return self._final_summary

            return (
                f"cycle {self._cycle_num} (target {self._target_idx + 1}"
                f"/{len(self._planned_path)}): "
                f"{', '.join(executed_summaries)} -> "
                f"distance to goal {final_dist:.2f} m"
            )

        except Exception as exc:  # noqa: BLE001
            self._completed = True
            self._final_summary = (
                f"Cycle failed: {type(exc).__name__}: {exc}"
            )
            return self._final_summary

    def _auto_advance_target_idx(self) -> None:
        """Advance ``self._target_idx`` past every waypoint within
        ``WAYPOINT_ADVANCE_THRESHOLD_M`` of the robot. Always leaves at
        least the last waypoint as a target."""
        ctrl = get_controller()
        x, y, _ = ctrl.get_pose()
        last = len(self._planned_path) - 1
        while self._target_idx < last:
            tx, ty = self._planned_path[self._target_idx]
            if math.hypot(tx - x, ty - y) < WAYPOINT_ADVANCE_THRESHOLD_M:
                self._target_idx += 1
            else:
                break

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def goal_reached_in_last_step(self) -> bool:
        return self._goal_marker

    @property
    def terminated(self) -> bool:
        return self._completed

    @property
    def run_dir(self) -> str | None:
        return str(self._run_dir) if self._run_dir else None


def build_agent(
    provider: str | None = None,
    model_name: str | None = None,
) -> Flow5Agent:
    """Build and return a flow_5 ``Flow5Agent``."""
    return Flow5Agent(provider=provider, model_name=model_name)
