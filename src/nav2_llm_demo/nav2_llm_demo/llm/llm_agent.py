"""LangGraph agent that navigates a robot using vision + tool calling.

The agent receives an annotated top-down map image showing the robot's
position, heading, and destination.  It calls movement tools (move_forward,
rotate) to drive the robot from source to destination.

A custom agent loop is used instead of create_react_agent so that map images
can be injected into the conversation as multimodal HumanMessages -- this is
required for the vision LLM to actually *see* the map.
"""

import base64
import json
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

from langchain.chat_models import init_chat_model
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from .map_renderer import render_annotated_map, render_full_map


# ---------------------------------------------------------------------------
# Protocol that the ROS node must implement
# ---------------------------------------------------------------------------

class RobotController(Protocol):
    """Interface the agent tools call into.  Implemented by the ROS node."""

    def get_pose(self) -> tuple[float, float, float]:
        ...

    def move_forward(self, distance_m: float, speed: float = 0.15) -> str:
        ...

    def rotate(self, angle_deg: float, speed: float = 0.5) -> str:
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


# ---------------------------------------------------------------------------
# Module-level controller reference
# ---------------------------------------------------------------------------

_controller: RobotController | None = None


def set_controller(ctrl: RobotController) -> None:
    global _controller
    _controller = ctrl


def _ctrl() -> RobotController:
    if _controller is None:
        raise RuntimeError("RobotController not registered; call set_controller first")
    return _controller


# ---------------------------------------------------------------------------
# Tool definitions (as dicts for bind_tools, not @tool decorators)
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
                        "description": "Distance in meters (positive=forward, negative=backward).",
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
                "Positive = counter-clockwise (left), negative = clockwise (right). "
                "Example: rotate(90) turns left, rotate(-90) turns right."
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
                "(red arrow), destination (green circle), and source (blue circle). "
                "Call this FIRST and after every few moves to check progress."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_robot_pose",
            "description": "Get the robot's current x, y (meters) and yaw (degrees) in the map frame.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_goal_reached",
            "description": "Check whether the robot is close enough to the destination. Returns REACHED or NOT_REACHED.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are controlling a TurtleBot3 robot navigating a 2D occupancy-grid map.

Your task: move the robot from its current position to the **destination** \
(marked with a green plus / circle on the map).

HARD RULE — read this twice:
- BLACK pixels are obstacles. The robot is NOT allowed to enter, cross, \
clip, or even touch a black region. Plan paths that stay strictly on \
WHITE pixels. If the only direction toward the goal is blocked by black, \
take a longer detour through white corridors instead.
- Every tool call you make must be one of: move_forward, rotate, \
get_map_view, get_robot_pose, check_goal_reached. You must always pick \
one of these on every turn (no free text answers).

Map legend (you will be reminded of this on every image):
- BLUE circle  = start position (where the run began).
- RED dot      = current robot position. The small RED arrow on the dot \
points in the robot's facing direction (its heading / yaw).
- GREEN plus / circle = destination you must reach.
- WHITE pixels = free space the robot may drive on.
- BLACK pixels = walls / obstacles. NEVER drive into these.
- GRAY pixels  = unknown.

Tools:
- get_map_view(): See an annotated top-down map. Call this FIRST and \
after every few moves so you can re-verify position vs obstacles.
- move_forward(distance_meters): Drive forward (positive) or backward \
(negative). Use small steps (0.3-1.0 m).
- rotate(angle_degrees): Turn in place. Positive = left/CCW, negative = \
right/CW.
- get_robot_pose(): Get exact coordinates and heading as JSON.
- check_goal_reached(): Check if you have arrived.

Strategy:
1. Call get_map_view() to see the map and your position.
2. Look at the WHITE corridors. Trace a route from the RED dot to the \
GREEN destination that stays entirely inside WHITE.
3. Rotate to face the next white waypoint along that route.
4. Move forward in small steps (0.3-1.0 m), then re-check the map.
5. If the next forward step would enter or graze BLACK, do NOT issue \
move_forward. Rotate to find another white direction first.
6. When close, call check_goal_reached() to verify arrival.

Important:
- The map image uses standard orientation: +X = RIGHT, +Y = UP.
- Your heading (yaw) is measured counter-clockwise from +X axis.
- Dark pixels = walls. White = free space. Gray = unknown.
- Take small steps and check the map often to stay safe.
"""

# ---------------------------------------------------------------------------
# Per-image context message
#
# The OpenAI API does not "remember" why we sent an image; the model only
# sees what's literally in the conversation. To keep the navigation context
# fresh on every image (system prompts get diluted as the chat grows), we
# attach the same legend + safety rules with every map view we send.
# ---------------------------------------------------------------------------

MAP_IMAGE_CONTEXT = (
    "Here is the current map view. READ THIS LEGEND BEFORE DECIDING:\n"
    "- BLUE circle  = where I (the robot) started.\n"
    "- RED dot      = where I am right now. The small RED arrow on the dot "
    "is the direction I am currently facing (my heading).\n"
    "- GREEN plus / circle = my destination — I must reach this point.\n"
    "- WHITE pixels = free space. I am ALLOWED to drive on white.\n"
    "- BLACK pixels = obstacles / walls. I am STRICTLY FORBIDDEN from "
    "entering or crossing any black region. Going into black is failure.\n"
    "- GRAY pixels  = unknown; treat as unsafe.\n\n"
    "Decide my next action using ONLY this image and tool history. Your "
    "next reply MUST be exactly one tool call: move_forward, rotate, "
    "get_map_view, get_robot_pose, or check_goal_reached. Pick the "
    "action that moves the RED dot closer to the GREEN destination "
    "WITHOUT ever stepping into BLACK. If the straight line toward the "
    "goal crosses black, pick a rotate / detour through white instead."
)


# ---------------------------------------------------------------------------
# Agent class
# ---------------------------------------------------------------------------

class VisionNavigationAgent:
    """Custom agent loop with vision support for map images."""

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
        self._llm_with_tools = self._llm.bind_tools(TOOL_SCHEMAS)
        self._messages: list[Any] = []
        self._step_num = 0
        self._run_dir: Path | None = None
        self._llm_call_num = 0

    def _ensure_run_dir(self) -> Path:
        """Create and return the output directory for this run's debug images."""
        if self._run_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            workspace = os.environ.get("WORKSPACE_DIR", "/workspace")
            self._run_dir = Path(workspace) / "llm_agent_runs" / timestamp
            self._run_dir.mkdir(parents=True, exist_ok=True)
        return self._run_dir

    def _save_step_image(self, tool_result: str) -> None:
        """Save the map image and metadata for this step. This is only used for debug/log purposes."""

        img_b64 = self.get_map_b64()

        run_dir = self._ensure_run_dir()
        prefix = f"step_{self._step_num:03d}"

        png_path = run_dir / f"{prefix}_map.png"
        png_path.write_bytes(base64.b64decode(img_b64))

        meta_path = run_dir / f"{prefix}_meta.txt"
        meta_lines = [
            f"Step: {self._step_num}",
            f"Tool result: {tool_result}",
            "",
        ]

        # Include the LLM's reasoning from the last AI message
        for msg in reversed(self._messages):
            content = getattr(msg, "content", "")
            role = getattr(msg, "type", "")
            if role == "ai" and isinstance(content, str) and content.strip():
                meta_lines.append(f"LLM reasoning: {content}")
                break

        # Include tool calls the LLM made
        for msg in reversed(self._messages):
            calls = getattr(msg, "tool_calls", None)
            if calls:
                for tc in calls:
                    meta_lines.append(
                        f"Tool call: {tc['name']}({json.dumps(tc.get('args', {}))})"
                    )
                break

        meta_path.write_text("\n".join(meta_lines))

    def initialize(self, source_x: float, source_y: float, dest_x: float, dest_y: float) -> None:
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

        self._save_step_image(",\n".join(summaries))

        return " | ".join(summaries)

    def get_map_b64(self):
        ctrl = _ctrl()
        x, y, yaw = ctrl.get_pose()
        img_b64 = render_annotated_map(
            map_yaml_path=ctrl.map_yaml_path,
            robot_x=x, robot_y=y, robot_yaw=yaw,
            dest_x=ctrl.dest_x, dest_y=ctrl.dest_y,
            source_x=ctrl.source_x, source_y=ctrl.source_y,
            crop_radius_m=18.0,
            output_size=512,
        )
        return img_b64


    def _execute_tool(self, name: str, args: dict[str, Any]) -> tuple[str, str | None]:
        """Run a tool and return (text_result, base64_image_or_None)."""
        ctrl = _ctrl()

        if name == "move_forward":
            result = ctrl.move_forward(args.get("distance_meters", 0.0))
            return result, None

        if name == "rotate":
            result = ctrl.rotate(args.get("angle_degrees", 0.0))
            return result, None

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
            return json.dumps({"x": round(x, 4), "y": round(y, 4), "yaw_degrees": round(math.degrees(yaw), 2)}), None

        if name == "check_goal_reached":
            x, y, _ = ctrl.get_pose()
            dist = math.hypot(ctrl.dest_x - x, ctrl.dest_y - y)
            if dist < 0.5:
                return f"GOAL REACHED - {dist:.2f} meters from goal. Navigation complete!", None
            return f"NOT_REACHED - {dist:.2f} meters remaining to destination ({ctrl.dest_x:.2f}, {ctrl.dest_y:.2f}).", None

        return f"Unknown tool: {name}", None

    # ------------------------------------------------------------------
    # LLM-call logging
    # ------------------------------------------------------------------

    def _llm_call_dir(self) -> Path:
        """Folder for the *upcoming* LLM call (created on demand)."""
        run_dir = self._ensure_run_dir()
        d = run_dir / f"llm_controls_call_{self._llm_call_num:03d}"
        d.mkdir(parents=True, exist_ok=True)
        return d

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
                        "image_url": {"url": "<inline PNG, see image_sent.png>"},
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
            (call_dir / "image_sent.png").write_bytes(base64.b64decode(img_b64))

        request_payload = {
            "llm_call_num": self._llm_call_num,
            "agent_step": self._step_num,
            "model": getattr(self._llm, "model_name", None)
                or os.environ.get("LLM_MODEL", ""),
            "provider": os.environ.get("LLM_PROVIDER", ""),
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

    @property
    def goal_reached_in_last_step(self) -> bool:
        """Check if the last tool response indicated goal reached."""
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
    """Build and return a VisionNavigationAgent."""
    return VisionNavigationAgent(provider=provider, model_name=model_name)
