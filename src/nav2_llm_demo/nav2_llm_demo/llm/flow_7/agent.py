"""Flow 7 planner: deterministic graph execution + LLM route selection.

This is the route-planning agent originally developed on the
``btfarre2/flow6`` branch. It has been re-numbered to flow 7 here because
flow 6 in this branch is the pure-Nav2 baseline.

Unlike flows 1-5 (LLM tool-loop, custom motion) and flow 6 (pure Nav2,
no LLM), this agent is consumed by ``llm_route_agent_node`` and exposes
``plan(...)`` rather than the ``initialize/step`` tool-loop surface. The
ROS node builds a topology graph from the occupancy map, asks the LLM
which node path to follow, and drives each edge with a hand-rolled
rotate/forward controller that marks edges blocked and re-plans on
failure.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage

from ..controller import make_run_dir, resolve_llm_config
from ..request_logger import LlmRequestLogger


@dataclass
class PlannerDecision:
    """Structured route-planning output consumed by the ROS node."""

    path_nodes: list[str]
    notes: str
    raw_response: str


SYSTEM_PROMPT = """\
You are the route planner for a TurtleBot3 navigating on a deterministic topology graph.

You do not choose low-level movements. The robot already knows how to:
- rotate toward a target node
- move in short bounded steps
- detect when an attempted edge traversal is blocked

You only choose a legal route through graph nodes.

Return exactly one JSON object:
{
  "path_nodes": ["start", "anchor_2", "anchor_5", "goal"],
  "notes": "short one-sentence explanation"
}

Rules:
- Output JSON only.
- `path_nodes` must be a legal node path through the provided graph.
- The first node must be the provided current node.
- The last node must be the provided goal node.
- Never include an edge that is marked `blocked`.
- Prefer shorter legal routes unless the context says a route recently failed.
- Keep notes short.
"""


class RoutePlanningAgent:
    """Event-driven graph route planner."""

    def __init__(self, provider: str | None = None, model_name: str | None = None):
        if provider is None or model_name is None:
            resolved_provider, resolved_model = resolve_llm_config("7")
            provider = provider or resolved_provider
            model_name = model_name or resolved_model

        self._provider = provider
        self._model_name = model_name
        self._llm = init_chat_model(
            model=model_name,
            model_provider=provider,
            temperature=0.1,
        )
        self._run_dir: Path | None = None
        self._plan_num = 0

    def _ensure_run_dir(self) -> Path:
        if self._run_dir is None:
            self._run_dir = make_run_dir(flow="7")
        return self._run_dir

    def plan(self, planning_context: dict[str, Any], *, reason: str) -> PlannerDecision:
        self._plan_num += 1
        prompt = (
            f"Planning reason: {reason}\n"
            f"Planning context JSON:\n{json.dumps(planning_context, separators=(',', ':'))}\n\n"
            "Return the JSON route now."
        )

        request_logger = LlmRequestLogger(self._ensure_run_dir())
        request_logger.log_request(
            prefix=f"plan_{self._plan_num:03d}",
            system_prompt=SYSTEM_PROMPT,
            user_prompt=prompt,
            planning_context=planning_context,
            reason=reason,
            img_b64=None,
        )

        response = self._llm.invoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ])
        raw_text = self._coerce_text(response.content)
        decision = self._parse_decision(raw_text)
        self._save_debug_artifacts(reason, planning_context, raw_text)
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
        candidate = raw_text.strip()
        match = re.search(r"\{.*\}", candidate, re.DOTALL)
        if match:
            candidate = match.group(0)

        path_nodes: list[str] = []
        notes = raw_text.strip()

        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            parsed = None

        if isinstance(parsed, dict):
            raw_nodes = parsed.get("path_nodes", [])
            if isinstance(raw_nodes, list):
                path_nodes = [str(node_id).strip() for node_id in raw_nodes if str(node_id).strip()]
            elif isinstance(raw_nodes, str) and raw_nodes.strip():
                path_nodes = [raw_nodes.strip()]
            notes = str(parsed.get("notes", notes)).strip() or notes

        return PlannerDecision(
            path_nodes=path_nodes,
            notes=notes,
            raw_response=raw_text,
        )

    def _save_debug_artifacts(
        self,
        reason: str,
        planning_context: dict[str, Any],
        raw_text: str,
    ) -> None:
        run_dir = self._ensure_run_dir()
        prefix = f"plan_{self._plan_num:03d}"
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
) -> RoutePlanningAgent:
    return RoutePlanningAgent(provider=provider, model_name=model_name)
