"""Flow 1 agent: hand-rolled loop with multimodal HumanMessage injection.

Behavior is identical to the original ``llm_agent.py``, except:

- Per-step debug files (``step_NNN_map.png`` / ``step_NNN_meta.txt``) are
  no longer written. Only ``llm_controls_call_NNN/`` folders land on disk.
- The run directory now lives under
  ``<workspace>/llm_agent_runs/flow_1/<timestamp>/`` instead of mixing
  flows together at ``llm_agent_runs/<timestamp>/``.

The shared ``RobotController`` Protocol and registry have moved into
``..controller``; this module imports from there.
"""

from __future__ import annotations

import base64
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
from ..map_renderer import render_annotated_map
from .prompt import MAP_IMAGE_CONTEXT, SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Tool definitions (raw schemas for bind_tools)
# ---------------------------------------------------------------------------

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "move_forward",
            "description": (
                "Move the robot forward by the given number of meters. "
                "Use positive values to go forward, negative to go backward. "
                "Typical step size is 0.3 to 1.0 meters."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "distance_meters": {
                        "type": "number",
                        "description":
                            "Distance in meters (positive=forward, "
                            "negative=backward).",
                    },
                },
                "required": ["distance_meters"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rotate",
            "description": (
                "Rotate the robot in place by the given angle in degrees. "
                "Positive = counter-clockwise (left), negative = clockwise "
                "(right). Example: rotate(90) turns left, rotate(-90) "
                "turns right."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "angle_degrees": {
                        "type": "number",
                        "description": "Rotation angle in degrees.",
                    },
                },
                "required": ["angle_degrees"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_map_view",
            "description": (
                "Capture an annotated top-down map image showing the robot "
                "(red arrow), destination (green circle), and source (blue "
                "circle). Call this FIRST and after every few moves to "
                "check progress."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_robot_pose",
            "description":
                "Get the robot's current x, y (meters) and yaw (degrees) "
                "in the map frame.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_goal_reached",
            "description":
                "Check whether the robot is close enough to the destination. "
                "Returns REACHED or NOT_REACHED.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


# ---------------------------------------------------------------------------
# Agent class
# ---------------------------------------------------------------------------

class VisionNavigationAgent:
    """Custom agent loop with vision support for map images (flow 1)."""

    def __init__(
        self,
        provider: str | None = None,
        model_name: str | None = None,
    ) -> None:
        if provider is None or model_name is None:
            resolved_provider, resolved_model = resolve_llm_config("1")
            provider = provider or resolved_provider
            model_name = model_name or resolved_model

        self._provider = provider
        self._model_name = model_name

        self._llm = init_chat_model(
            model=model_name,
            model_provider=provider,
            temperature=0.1,
        )
        self._llm_with_tools = self._llm.bind_tools(TOOL_SCHEMAS)
        self._messages: list[Any] = []
        self._step_num = 0
        self._run_dir: Path | None = None
        self._llm_call_num = 0

    # ------------------------------------------------------------------
    # Run / log directory helpers
    # ------------------------------------------------------------------

    def _ensure_run_dir(self) -> Path:
        if self._run_dir is None:
            self._run_dir = make_run_dir(flow="1")
        return self._run_dir

    def _llm_call_dir(self) -> Path:
        run_dir = self._ensure_run_dir()
        d = run_dir / f"llm_controls_call_{self._llm_call_num:03d}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    # ------------------------------------------------------------------
    # Conversation management
    # ------------------------------------------------------------------

    def initialize(
        self,
        source_x: float,
        source_y: float,
        dest_x: float,
        dest_y: float,
    ) -> None:
        """Set up the initial conversation."""
        user_msg = (
            f"Navigate the robot to the destination.\n"
            f"Source (start): ({source_x:.2f}, {source_y:.2f})\n"
            f"Destination (goal): ({dest_x:.2f}, {dest_y:.2f})\n\n"
            f"Begin by calling get_map_view() to see the map."
        )
        self._messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_msg),
        ]

    def step(self) -> str:
        """Run one agent turn: LLM call -> tool execution -> inject results.

        May involve multiple tool calls in one turn (parallel tool calling).
        Returns a summary of what happened.
        """
        self._step_num += 1
        self._log_llm_call(self._messages)
        response = self._llm_with_tools.invoke(self._messages)
        self._log_llm_response(response)
        self._messages.append(response)

        tool_calls = getattr(response, "tool_calls", None) or []
        if not tool_calls:
            return f"Agent message (no tool calls): {response.content}"

        summaries: list[str] = []

        for tc in tool_calls:
            name = tc["name"]
            args = tc.get("args", {})
            tc_id = tc["id"]

            text_result, img_b64 = self._execute_tool(name, args)
            summaries.append(f"{name}: {text_result}")

            self._messages.append(
                ToolMessage(content=text_result, tool_call_id=tc_id)
            )

            if img_b64 is not None:
                self._messages.append(
                    HumanMessage(content=[
                        {"type": "text", "text": MAP_IMAGE_CONTEXT},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{img_b64}",
                            },
                        },
                    ])
                )

        return " | ".join(summaries)

    # ------------------------------------------------------------------
    # Map rendering
    # ------------------------------------------------------------------

    def get_map_b64(self) -> str:
        ctrl = get_controller()
        x, y, yaw = ctrl.get_pose()
        return render_annotated_map(
            map_yaml_path=ctrl.map_yaml_path,
            robot_x=x, robot_y=y, robot_yaw=yaw,
            dest_x=ctrl.dest_x, dest_y=ctrl.dest_y,
            source_x=ctrl.source_x, source_y=ctrl.source_y,
            crop_radius_m=18.0,
            output_size=512,
        )

    # ------------------------------------------------------------------
    # Tool dispatch
    # ------------------------------------------------------------------

    def _execute_tool(
        self, name: str, args: dict[str, Any],
    ) -> tuple[str, str | None]:
        """Run a tool and return (text_result, base64_image_or_None)."""
        ctrl = get_controller()

        if name == "move_forward":
            return ctrl.move_forward(args.get("distance_meters", 0.0)), None

        if name == "rotate":
            return ctrl.rotate(args.get("angle_degrees", 0.0)), None

        if name == "get_map_view":
            img_b64 = self.get_map_b64()
            text = (
                "Map view captured. Wait for the next human message: it "
                "will contain the map image and instructions on what to "
                "do next."
            )
            return text, img_b64

        if name == "get_robot_pose":
            x, y, yaw = ctrl.get_pose()
            return json.dumps({
                "x": round(x, 4),
                "y": round(y, 4),
                "yaw_degrees": round(math.degrees(yaw), 2),
            }), None

        if name == "check_goal_reached":
            x, y, _ = ctrl.get_pose()
            dist = math.hypot(ctrl.dest_x - x, ctrl.dest_y - y)
            if dist < 0.5:
                return (
                    f"GOAL REACHED - {dist:.2f} meters from goal. "
                    "Navigation complete!",
                    None,
                )
            return (
                f"NOT_REACHED - {dist:.2f} meters remaining to "
                f"destination ({ctrl.dest_x:.2f}, {ctrl.dest_y:.2f}).",
                None,
            )

        return f"Unknown tool: {name}", None

    # ------------------------------------------------------------------
    # LLM-call logging
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_last_image_b64(messages: list[Any]) -> str | None:
        """Return base64 of the most recent inline PNG in the message list."""
        for msg in reversed(messages):
            content = getattr(msg, "content", None)
            if not isinstance(content, list):
                continue
            for part in reversed(content):
                if not isinstance(part, dict):
                    continue
                if part.get("type") != "image_url":
                    continue
                url = (part.get("image_url") or {}).get("url", "")
                if "base64," in url:
                    return url.split("base64,", 1)[1]
        return None

    @staticmethod
    def _message_to_dict(msg: Any) -> dict[str, Any]:
        """Flatten one LangChain message into a JSON-friendly dict.

        Inline image data URLs are replaced with a placeholder so the JSON
        stays small; the actual bytes are written separately to
        ``image_sent.png``.
        """
        out: dict[str, Any] = {"role": type(msg).__name__}

        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls:
            out["tool_calls"] = tool_calls

        tool_call_id = getattr(msg, "tool_call_id", None)
        if tool_call_id:
            out["tool_call_id"] = tool_call_id

        content = getattr(msg, "content", None)
        if isinstance(content, str):
            out["content"] = content
        elif isinstance(content, list):
            parts: list[dict[str, Any]] = []
            for part in content:
                if not isinstance(part, dict):
                    parts.append({"raw": str(part)[:500]})
                    continue
                if part.get("type") == "image_url":
                    parts.append({
                        "type": "image_url",
                        "image_url": {
                            "url": "<inline PNG, see image_sent.png>",
                        },
                    })
                else:
                    parts.append(part)
            out["content"] = parts
        elif content is not None:
            out["content"] = str(content)[:2000]

        return out

    def _log_llm_call(self, messages: list[Any]) -> None:
        """Persist the exact request that is about to be sent to the LLM."""
        self._llm_call_num += 1
        call_dir = self._llm_call_dir()

        img_b64 = self._extract_last_image_b64(messages)
        if img_b64:
            (call_dir / "image_sent.png").write_bytes(
                base64.b64decode(img_b64)
            )

        request_payload = {
            "llm_call_num": self._llm_call_num,
            "agent_step": self._step_num,
            "model": self._model_name,
            "provider": self._provider,
            "image_sent": "image_sent.png" if img_b64 else None,
            "messages": [self._message_to_dict(m) for m in messages],
        }
        (call_dir / "request.json").write_text(
            json.dumps(request_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _log_llm_response(self, response: Any) -> None:
        """Persist the LLM response right after invoke returns."""
        call_dir = self._llm_call_dir()
        content = getattr(response, "content", "")
        if not isinstance(content, str):
            content = str(content)
        out = {
            "llm_call_num": self._llm_call_num,
            "role": type(response).__name__,
            "content": content,
            "tool_calls": getattr(response, "tool_calls", None) or [],
        }
        (call_dir / "response.json").write_text(
            json.dumps(out, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def terminated(self) -> bool:
        """flow_1 has no terminal state of its own; each ``step()`` is one
        LLM round-trip and the loop is expected to keep going. The ROS-side
        loop only bails on this when an agent self-reports done."""
        return False

    @property
    def goal_reached_in_last_step(self) -> bool:
        """True if the last few messages contain a GOAL REACHED marker."""
        for msg in reversed(self._messages[-5:]):
            content = getattr(msg, "content", "")
            if isinstance(content, str) and "GOAL REACHED" in content:
                return True
        return False

    @property
    def run_dir(self) -> str | None:
        """Path to the debug output directory for this run, or None."""
        return str(self._run_dir) if self._run_dir else None


def build_agent(
    provider: str | None = None,
    model_name: str | None = None,
) -> VisionNavigationAgent:
    """Build and return a flow 1 ``VisionNavigationAgent``."""
    return VisionNavigationAgent(provider=provider, model_name=model_name)
