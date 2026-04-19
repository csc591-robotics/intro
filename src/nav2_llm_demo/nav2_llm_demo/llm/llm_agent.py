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

    route_mode: str
    preferred_recovery_side: str
    avoid_regions: list[str]
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
  "route_mode": "continue_heading | prefer_left_open_region | prefer_right_open_region | backtrack_then_replan | avoid_failed_region",
  "preferred_recovery_side": "left | right | none",
  "avoid_regions": ["optional short region identifiers"],
  "notes": "short one-sentence explanation"
}

Rules:
- Output JSON only. Do not wrap it in markdown fences.
- Prefer short notes.
- Use failed-region summaries to avoid retrying locally bad areas.
- Use the map image and structured context together.
- If the robot is simply on a workable route, use "continue_heading".
- If one side is clearly a better escape branch, express that in both
  "route_mode" and "preferred_recovery_side".
- Use "backtrack_then_replan" if the robot should retreat before trying a new route.
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
        route_mode = "continue_heading"
        preferred_side = "none"
        avoid_regions: list[str] = []
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
            route_mode = str(parsed.get("route_mode", route_mode))
            preferred_side = str(parsed.get("preferred_recovery_side", preferred_side))
            avoid_regions_raw = parsed.get("avoid_regions", [])
            if isinstance(avoid_regions_raw, list):
                avoid_regions = [str(item) for item in avoid_regions_raw if str(item).strip()]
            notes = str(parsed.get("notes", notes)).strip() or notes
        else:
            lowered = raw_text.lower()
            if "backtrack" in lowered:
                route_mode = "backtrack_then_replan"
            elif "avoid_failed_region" in lowered or "avoid region" in lowered:
                route_mode = "avoid_failed_region"
            elif "left" in lowered:
                route_mode = "prefer_left_open_region"
                preferred_side = "left"
            elif "right" in lowered:
                route_mode = "prefer_right_open_region"
                preferred_side = "right"

        allowed_modes = {
            "continue_heading",
            "prefer_left_open_region",
            "prefer_right_open_region",
            "backtrack_then_replan",
            "avoid_failed_region",
        }
        if route_mode not in allowed_modes:
            route_mode = "continue_heading"

        if preferred_side not in {"left", "right", "none"}:
            preferred_side = "none"

        if route_mode == "prefer_left_open_region":
            preferred_side = "left"
        elif route_mode == "prefer_right_open_region":
            preferred_side = "right"

        return PlannerDecision(
            route_mode=route_mode,
            preferred_recovery_side=preferred_side,
            avoid_regions=avoid_regions,
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
