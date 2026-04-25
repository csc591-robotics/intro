"""Flow_4: fixed-topology gather -> decide -> execute -> check_goal cycle.

This is NOT a ReAct agent. The graph is hard-coded in plain Python: each
``step()`` runs exactly one cycle and the LLM only ever picks between
``move_forward`` and ``rotate``. No "look without acting" is possible.

Cycle layout (see ``__init__`` docstring for the diagram):

    1. gather_situation  (Python; no LLM)
    2. decide_action     (LLM; tools = [move_forward, rotate], tool_choice=any)
    3. execute_action    (Python; runs the chosen tool via the controller)
    4. check_goal        (Python; goal reached if dist to dest < 1.0 m)

State surface (matches the other flows so the ROS node is unchanged):

* ``initialize(source_x, source_y, dest_x, dest_y)``
* ``step()`` -> str
* ``goal_reached_in_last_step`` (property)
* ``terminated`` (property; True after goal reached or unrecoverable error)
* ``run_dir`` (property; for log discovery)

Same ``PerCallLogger`` (from flow_2) is used for the per-LLM-call
artifacts, and the same ``compact_history`` + ``prune_old_images``
pipeline keeps the request size under the rate-limit budget.
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
from .prompt import DECIDE_SYSTEM_PROMPT
from .situation import build_situation_message


GOAL_REACHED_THRESHOLD_M = 1.0


_DECIDE_TOOLS = [move_forward, rotate]


class Flow4Agent:
    """Fixed-cycle vision + LiDAR navigation agent."""

    def __init__(
        self,
        provider: str | None = None,
        model_name: str | None = None,
    ) -> None:
        if provider is None or model_name is None:
            resolved_provider, resolved_model = resolve_llm_config("4")
            provider = provider or resolved_provider
            model_name = model_name or resolved_model

        self._provider = provider
        self._model_name = model_name

        base_llm = init_chat_model(
            model=model_name,
            model_provider=provider,
            temperature=0.0,
        )
        # tool_choice="any" forces the LLM to emit a tool call (vs free
        # text). That's the whole point of flow_4: no "thinking out loud",
        # every cycle ends in exactly one move_forward or rotate.
        try:
            self._llm = base_llm.bind_tools(
                _DECIDE_TOOLS, tool_choice="any",
            )
        except TypeError:
            # Older provider integrations may not accept tool_choice.
            self._llm = base_llm.bind_tools(_DECIDE_TOOLS)

        self._messages: list[Any] = []
        self._cycle_num = 0
        self._run_dir: Path | None = None
        self._logger: PerCallLogger | None = None
        self._completed = False
        self._final_summary: str | None = None
        self._dest_x = 0.0
        self._dest_y = 0.0
        self._goal_marker = False

    # ------------------------------------------------------------------
    # Run / log directory
    # ------------------------------------------------------------------

    def _ensure_run_dir(self) -> Path:
        if self._run_dir is None:
            self._run_dir = make_run_dir(flow="4")
            self._logger = PerCallLogger(
                self._run_dir,
                provider=self._provider,
                model=self._model_name,
            )
        return self._run_dir

    # ------------------------------------------------------------------
    # Conversation setup
    # ------------------------------------------------------------------

    def initialize(
        self,
        source_x: float,
        source_y: float,
        dest_x: float,
        dest_y: float,
    ) -> None:
        self._ensure_run_dir()
        self._dest_x = dest_x
        self._dest_y = dest_y
        # Preamble: a short HumanMessage that compact_history will preserve
        # at index 0 across many cycles. It fixes the goal coordinates so
        # the LLM never forgets the destination after pruning.
        self._messages = [
            HumanMessage(content=(
                f"Goal: navigate from your current position to "
                f"({dest_x:.2f}, {dest_y:.2f}) in the map frame. "
                "Each turn I will provide a fresh map+LiDAR situation; "
                "respond with EXACTLY ONE move_forward or rotate "
                "tool call. Goal-reached threshold: 1.0 m."
            )),
        ]
        self._cycle_num = 0
        self._completed = False
        self._final_summary = None
        self._goal_marker = False

    # ------------------------------------------------------------------
    # One cycle
    # ------------------------------------------------------------------

    def step(self) -> str:
        if self._completed:
            return self._final_summary or "Agent already completed."

        self._cycle_num += 1
        self._ensure_run_dir()

        try:
            # 1. GATHER ----------------------------------------------------
            situation_msg, _artifact = build_situation_message()
            self._messages.append(situation_msg)

            # 2. DECIDE ----------------------------------------------------
            #    Trim history before the LLM call to stay under the
            #    rate-limit budget. compact_history preserves the
            #    preamble (messages[0]) and the last 4 action rounds;
            #    prune_old_images strips stale base64 PNGs.
            compacted = compact_history(self._messages, keep_rounds=4)
            pruned = prune_old_images(compacted, keep_last=1)
            llm_input = [SystemMessage(content=DECIDE_SYSTEM_PROMPT)] + pruned

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
                    result = get_controller().move_forward(
                        float(args.get("distance_meters", 0.0))
                    )
                elif name == "rotate":
                    result = get_controller().rotate(
                        float(args.get("angle_degrees", 0.0))
                    )
                else:
                    result = (
                        f"Tool {name!r} is not allowed in flow_4. Only "
                        "move_forward and rotate are bound."
                    )
                executed_summaries.append(f"{name}({json.dumps(args)})")
                self._messages.append(
                    ToolMessage(content=result, tool_call_id=tc_id)
                )

            # 4. CHECK GOAL ------------------------------------------------
            x, y, _ = get_controller().get_pose()
            dist = math.hypot(self._dest_x - x, self._dest_y - y)
            if dist < GOAL_REACHED_THRESHOLD_M:
                self._completed = True
                self._goal_marker = True
                self._final_summary = (
                    f"GOAL REACHED after {self._cycle_num} cycles "
                    f"({dist:.2f} m from destination). "
                    f"Last action: {', '.join(executed_summaries)}."
                )
                return self._final_summary

            return (
                f"cycle {self._cycle_num}: "
                f"{', '.join(executed_summaries)} -> "
                f"distance to goal {dist:.2f} m"
            )

        except Exception as exc:  # noqa: BLE001 - surface LLM/tool errors
            self._completed = True
            self._final_summary = (
                f"Cycle failed: {type(exc).__name__}: {exc}"
            )
            return self._final_summary

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
) -> Flow4Agent:
    """Build and return a flow_4 ``Flow4Agent``."""
    return Flow4Agent(provider=provider, model_name=model_name)
