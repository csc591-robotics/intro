"""LLM replanning helper for event-driven robot navigation.

The planner is only consulted on meaningful navigation events such as startup
and replanning after blocked or failed recovery sequences. Low-level movement
remains deterministic in the ROS node.
"""

import base64
import json
import math
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage

from .map_renderer import render_annotated_map
from .request_logger import LlmRequestLogger


class RobotController(Protocol):
    """Interface the planner uses to gather map-rendering context."""

    def get_pose(self) -> tuple[float, float, float]:
        ...

    def get_navigation_context(self) -> dict[str, Any]:
        ...

    @property
    def map_yaml_path(self) -> str: ...
    @property
    def source_x(self) -> float: ...
    @property
    def source_y(self) -> float: ...
    @property
    def dest_x(self) -> float: ...
    @property
    def dest_y(self) -> float: ...


@dataclass
class PlannerDecision:
    """Structured replanning output consumed by the ROS node."""

    route_sequence: list[str]
    notes: str
    raw_response: str


_controller: RobotController | None = None


def set_controller(ctrl: RobotController) -> None:
    global _controller
    _controller = ctrl


def _ctrl() -> RobotController:
    if _controller is None:
        raise RuntimeError("RobotController not registered; call set_controller first")
    return _controller


SYSTEM_PROMPT = """\
You are the global replanning module for a TurtleBot3 navigating a 2D occupancy-grid map.

You are not responsible for low-level motor actions. A deterministic local
controller handles:
- turning
- short forward motion
- backing up
- bounded recovery

You are responsible for deciding the high-level route intent when navigation is
starting or when the current local strategy has failed.

Return exactly one JSON object with this schema:
{
  "route_sequence": [
    "continue_route | take_left_branch | take_right_branch | backtrack"
  ],
  "notes": "short one-sentence explanation"
}

Rules:
- Output JSON only. Do not wrap it in markdown fences.
- Prefer short notes.
- Use failed-branch summaries to avoid retrying route choices that already failed.
- Use the map image and structured context together.
- Return a short route sequence of 1 to 4 actions.
- If the robot is simply on a workable route, use "continue_route".
- Use "take_left_branch" or "take_right_branch" only for route-level branch choices.
- Use "backtrack" only when the current route has failed and the robot should retreat.
- Do not choose recovery side, turn angle, step size, or any other low-level motion detail.
"""


class VisionNavigationAgent:
    """Event-driven planner that returns compact replanning decisions."""

    def __init__(self, provider: str | None = None, model_name: str | None = None):
        provider = provider or os.environ.get("LLM_PROVIDER", "")
        model_name = model_name or os.environ.get("LLM_MODEL", "")

        if not provider:
            raise RuntimeError("LLM_PROVIDER not set. Add it to your .env file.")
        if not model_name:
            raise RuntimeError("LLM_MODEL not set. Add it to your .env file.")

        self._llm = init_chat_model(
            model=model_name,
            model_provider=provider,
            temperature=0.1,
        )
        self._run_dir: Path | None = None
        self._plan_num = 0
        self._source: tuple[float, float] | None = None
        self._dest: tuple[float, float] | None = None

    def initialize(self, source_x: float, source_y: float, dest_x: float, dest_y: float) -> None:
        self._source = (source_x, source_y)
        self._dest = (dest_x, dest_y)

    def _ensure_run_dir(self) -> Path:
        if self._run_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            workspace = os.environ.get("WORKSPACE_DIR", "/workspace")
            self._run_dir = Path(workspace) / "llm_agent_runs" / timestamp
            self._run_dir.mkdir(parents=True, exist_ok=True)
        return self._run_dir

    def get_map_b64(self) -> str:
        ctrl = _ctrl()
        x, y, yaw = ctrl.get_pose()
        return render_annotated_map(
            map_yaml_path=ctrl.map_yaml_path,
            robot_x=x,
            robot_y=y,
            robot_yaw=yaw,
            dest_x=ctrl.dest_x,
            dest_y=ctrl.dest_y,
            source_x=ctrl.source_x,
            source_y=ctrl.source_y,
            crop_radius_m=10.0,
            output_size=512,
        )

    def plan(
        self,
        planning_context: dict[str, Any],
        *,
        reason: str,
        include_map: bool = True,
    ) -> PlannerDecision:
        """Return a structured high-level replanning decision."""
        self._plan_num += 1

        prompt = (
            f"Planning reason: {reason}\n"
            f"Source: {self._source}\n"
            f"Destination: {self._dest}\n"
            f"Planning context JSON:\n{json.dumps(planning_context, indent=2)}\n\n"
            "Return the JSON decision now."
        )

        messages: list[Any] = [SystemMessage(content=SYSTEM_PROMPT)]
        if include_map:
            img_b64 = self.get_map_b64()
            messages.append(
                HumanMessage(content=[
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img_b64}"},
                    },
                ])
            )
        else:
            img_b64 = None
            messages.append(HumanMessage(content=prompt))

        request_logger = LlmRequestLogger(self._ensure_run_dir())
        request_logger.log_request(
            prefix=f"plan_{self._plan_num:03d}",
            system_prompt=SYSTEM_PROMPT,
            user_prompt=prompt,
            planning_context=planning_context,
            reason=reason,
            img_b64=img_b64,
        )

        response = self._llm.invoke(messages)
        raw_text = self._coerce_text(response.content)
        decision = self._parse_decision(raw_text)
        self._save_debug_artifacts(reason, planning_context, raw_text, img_b64)
        return decision

    def _coerce_text(self, content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
            return "\n".join(parts)
        return str(content)

    def _parse_decision(self, raw_text: str) -> PlannerDecision:
        route_sequence = ["continue_route"]
        notes = raw_text.strip()

        parsed: dict[str, Any] | None = None
        candidate = raw_text.strip()
        match = re.search(r"\{.*\}", candidate, re.DOTALL)
        if match:
            candidate = match.group(0)
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            parsed = None

        if parsed is not None:
            raw_sequence = parsed.get("route_sequence", route_sequence)
            if isinstance(raw_sequence, list):
                route_sequence = [str(item) for item in raw_sequence if str(item).strip()]
            elif isinstance(raw_sequence, str) and raw_sequence.strip():
                route_sequence = [raw_sequence.strip()]
            notes = str(parsed.get("notes", notes)).strip() or notes
        else:
            lowered = raw_text.lower()
            inferred: list[str] = []
            if "backtrack" in lowered:
                inferred.append("backtrack")
            if "left" in lowered:
                inferred.append("take_left_branch")
            if "right" in lowered:
                inferred.append("take_right_branch")
            if not inferred:
                inferred.append("continue_route")
            route_sequence = inferred[:4]

        allowed_intents = {
            "continue_route",
            "take_left_branch",
            "take_right_branch",
            "backtrack",
        }
        cleaned_sequence = [intent for intent in route_sequence if intent in allowed_intents]
        if not cleaned_sequence:
            cleaned_sequence = ["continue_route"]
        cleaned_sequence = cleaned_sequence[:4]

        return PlannerDecision(
            route_sequence=cleaned_sequence,
            notes=notes,
            raw_response=raw_text,
        )

    def _save_debug_artifacts(
        self,
        reason: str,
        planning_context: dict[str, Any],
        raw_text: str,
        img_b64: str | None,
    ) -> None:
        run_dir = self._ensure_run_dir()
        prefix = f"plan_{self._plan_num:03d}"

        if img_b64 is not None:
            png_path = run_dir / f"{prefix}_map.png"
            png_path.write_bytes(base64.b64decode(img_b64))

        meta_path = run_dir / f"{prefix}_meta.txt"
        meta_path.write_text(
            "\n".join(
                [
                    f"Plan number: {self._plan_num}",
                    f"Reason: {reason}",
                    "",
                    "Planning context:",
                    json.dumps(planning_context, indent=2),
                    "",
                    "LLM response:",
                    raw_text,
                ]
            )
        )

    @property
    def run_dir(self) -> str | None:
        return str(self._run_dir) if self._run_dir else None


def build_agent(
    provider: str | None = None,
    model_name: str | None = None,
) -> VisionNavigationAgent:
    return VisionNavigationAgent(provider=provider, model_name=model_name)
