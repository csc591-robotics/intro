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

from .map_renderer import render_annotated_map


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

    def get_navigation_context(self) -> dict[str, Any]:
        ...

    def turn_toward_goal(self) -> str:
        ...

    def advance_step(self) -> str:
        ...

    def recover_to_open_side(self) -> str:
        ...

    def back_up(self) -> str:
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
            "name": "turn_toward_goal",
            "description": (
                "Rotate the robot toward the goal using grounded heading logic. "
                "Use this instead of choosing small raw turn angles yourself."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "advance_step",
            "description": (
                "Advance toward the goal using the grounded controller. "
                "The controller chooses the exact safe forward step size."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recover_to_open_side",
            "description": (
                "Run a deterministic recovery maneuver toward the more open side. "
                "Use this after blocked moves or regressions instead of guessing a raw turn."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "back_up",
            "description": (
                "Reverse by a short deterministic amount to escape tight spaces."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_navigation_context",
            "description": (
                "Get grounded navigation context from the map and current pose: "
                "distance to goal, heading error, local obstacle clearance, "
                "blocked streak, regression streak, and recommended safe forward step."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_map_view",
            "description": (
                "Capture an annotated top-down map image showing the robot "
                "(red arrow), destination (green circle), and source (blue circle). "
                "Call this FIRST and after every few high-level actions to check progress."
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
You are the high-level planner for a TurtleBot3 navigating a 2D occupancy-grid map.

Your task: move the robot from its current position to the **destination** \
(marked with a green circle on the map).

Available tools:
- get_navigation_context(): Get grounded spatial signals from the map. Use this
before deciding what to do next.
- get_map_view(): See an annotated top-down map. The image will be shown to \
you. Red arrow = robot position and heading. Green circle = destination. \
Blue circle = start. Call this FIRST and after every few moves.
- turn_toward_goal(): Ask the controller to rotate toward the goal.
- advance_step(): Ask the controller to make one grounded forward advance.
- recover_to_open_side(): Ask the controller to perform a deterministic local
recovery toward the more open side.
- back_up(): Ask the controller to reverse a short amount.
- get_robot_pose(): Get exact coordinates and heading as JSON.
- check_goal_reached(): Check if you have arrived.

Strategy:
1. Call get_map_view() to see the map and your position.
2. Call get_navigation_context() to ground your next action in obstacle
clearance and heading error.
3. Decide the next high-level action instead of micromanaging geometry.
4. Use turn_toward_goal() when heading_error_deg is large.
5. Use advance_step() when the heading is acceptable and forward progress looks plausible.
6. If blocked or regressing, prefer recover_to_open_side() or back_up() over repeating the same failed action.
7. When close, call check_goal_reached() to verify arrival.
8. Prefer corridors and open paths. If a route keeps failing, switch strategy instead of dithering.

Important:
- The map image uses standard orientation: +X = RIGHT, +Y = UP.
- Your heading (yaw) is measured counter-clockwise from +X axis.
- Dark pixels = walls. White = free space. Gray = unknown.
- Do not try to control exact turn angles or exact move distances yourself.
- Do not attempt advance_step() when heading_error_deg is large.
- Respect recommended_step_m from get_navigation_context().
- Use recovery tools after blocked or regressive outcomes instead of retrying the same action.
"""


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
        response = self._llm_with_tools.invoke(self._messages)
        self._messages.append(response)

        # tools expected by agent
        tool_calls = getattr(response, "tool_calls", None) or []
        if not tool_calls:
            return f"Agent message (no tool calls): {response.content}"

        summaries: list[str] = []
        current_messages: list[HumanMessage] = []

        for tc in tool_calls:
            name = tc["name"]
            args = tc.get("args", {})
            tc_id = tc["id"]

            text_result, img_b64 = self._execute_tool(name, args)
            summaries.append(f"{name}: {text_result}")

            # ToolMessage records the textual result for a specific tool_call_id
            self._messages.append(
                ToolMessage(content=text_result, tool_call_id=tc_id)
            )


            if img_b64 is not None:
                # HumanMessage feeds rendered map to llm
                current_messages.append(
                    HumanMessage(content=[
                        {"type": "text", "text": "Here is the current map view:"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{img_b64}",
                            },
                        },
                    ])
                )
        
        # expects all tool calls + messages to be in same turn
        self._messages.extend(current_messages)
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
            crop_radius_m=10.0,
            output_size=512,
        )
        return img_b64


    def _execute_tool(self, name: str, args: dict[str, Any]) -> tuple[str, str | None]:
        """Run a tool and return (text_result, base64_image_or_None)."""
        ctrl = _ctrl()

        if name == "get_navigation_context":
            return json.dumps(ctrl.get_navigation_context()), None

        if name == "turn_toward_goal":
            result = ctrl.turn_toward_goal()
            return result, None

        if name == "advance_step":
            result = ctrl.advance_step()
            return result, None

        if name == "recover_to_open_side":
            result = ctrl.recover_to_open_side()
            return result, None

        if name == "back_up":
            result = ctrl.back_up()
            return result, None

        if name == "get_map_view":
            x, y, yaw = ctrl.get_pose()

            img_b64 = self.get_map_b64()

            dist = math.hypot(ctrl.dest_x - x, ctrl.dest_y - y)
            text = (
                f"Map view captured. Robot at ({x:.2f}, {y:.2f}), "
                f"heading {math.degrees(yaw):.1f} deg. "
                f"Distance to destination: {dist:.2f} m."
            )
            return text, img_b64

        if name == "get_robot_pose":
            x, y, yaw = ctrl.get_pose()
            return json.dumps({"x": round(x, 4), "y": round(y, 4), "yaw_degrees": round(math.degrees(yaw), 2)}), None

        if name == "check_goal_reached":
            x, y, _ = ctrl.get_pose()
            dist = math.hypot(ctrl.dest_x - x, ctrl.dest_y - y)
            if dist < 0.5:
                return f"REACHED - {dist:.2f} meters from goal. Navigation complete!", None
            return f"NOT_REACHED - {dist:.2f} meters remaining to destination ({ctrl.dest_x:.2f}, {ctrl.dest_y:.2f}).", None

        return f"Unknown tool: {name}", None

    @property
    def goal_reached_in_last_step(self) -> bool:
        """Check if the last tool response indicated goal reached."""
        for msg in reversed(self._messages[-5:]):
            content = getattr(msg, "content", "")
            if isinstance(content, str) and content.startswith("REACHED -"):
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
