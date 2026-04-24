"""Tool definitions for flow 2 (LangChain ``@tool`` decorators).

Movement tools and the goal-checking tool return plain strings; LangGraph's
``ToolNode`` wraps them in a normal ``ToolMessage(content=string)``.

``get_map_view`` returns a *list of content blocks* (text + image_url).
``ToolNode`` propagates that as ``ToolMessage(content=[...blocks...])``,
which the OpenAI / gpt-4o tool channel accepts as multimodal content. That
delivers the map image to the LLM inside the tool result itself, with no
extra ``HumanMessage`` injected by the agent loop.
"""

from __future__ import annotations

import json
import math

from langchain_core.tools import tool

from ..controller import get_controller
from ..map_renderer import render_annotated_map


# Reiterated on every map view so the legend stays adjacent to the image
# even after many turns dilute the system prompt.
MAP_TOOL_TEXT = (
    "Current map view. Legend:\n"
    "- BLUE circle  = where I started.\n"
    "- RED dot      = where I am right now. The small RED arrow on the dot "
    "is my heading (the direction I am facing).\n"
    "- GREEN plus / circle = my destination.\n"
    "- WHITE pixels = free space; I am ALLOWED to drive here.\n"
    "- BLACK pixels = walls / obstacles; I am STRICTLY FORBIDDEN from "
    "entering or crossing them. Driving into black is failure.\n"
    "- GRAY pixels  = unknown; treat as unsafe.\n"
    "Choose ONE next tool call (move_forward, rotate, get_map_view, "
    "get_robot_pose, or check_goal_reached) that moves the RED dot toward "
    "the GREEN destination without ever stepping into BLACK. After any "
    "movement, your next call MUST be get_map_view() again."
)


@tool
def move_forward(distance_meters: float) -> str:
    """Drive the robot forward (positive) or backward (negative) by the given
    number of meters. Use small steps (0.3 to 1.0 m). Robot stops on its
    own when the distance is reached.
    """
    ctrl = get_controller()
    return ctrl.move_forward(float(distance_meters))


@tool
def rotate(angle_degrees: float) -> str:
    """Rotate the robot in place. Positive = counter-clockwise (left),
    negative = clockwise (right). Robot stops on its own when the rotation
    is complete.
    """
    ctrl = get_controller()
    return ctrl.rotate(float(angle_degrees))


@tool(response_format="content_and_artifact")
def get_map_view() -> tuple[list[dict], dict]:
    """Return a fresh annotated top-down map view as a multimodal tool result.

    The first element is the LangChain content list (text + image_url) that
    becomes ``ToolMessage.content`` and is shown to the model. The second
    element (``artifact``) is metadata kept on the message but hidden from
    the LLM context, used by our PerCallLogger to extract the raw PNG.
    """
    ctrl = get_controller()
    x, y, yaw = ctrl.get_pose()
    img_b64 = render_annotated_map(
        map_yaml_path=ctrl.map_yaml_path,
        robot_x=x, robot_y=y, robot_yaw=yaw,
        dest_x=ctrl.dest_x, dest_y=ctrl.dest_y,
        source_x=ctrl.source_x, source_y=ctrl.source_y,
        crop_radius_m=18.0,
        output_size=512,
    )
    content: list[dict] = [
        {"type": "text", "text": MAP_TOOL_TEXT},
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{img_b64}",
            },
        },
    ]
    artifact = {"image_b64": img_b64}
    return content, artifact


@tool
def get_robot_pose() -> str:
    """Return the robot's current pose in the map frame as a JSON string with
    keys ``x``, ``y`` (meters) and ``yaw_degrees``.
    """
    ctrl = get_controller()
    x, y, yaw = ctrl.get_pose()
    return json.dumps({
        "x": round(x, 4),
        "y": round(y, 4),
        "yaw_degrees": round(math.degrees(yaw), 2),
    })


@tool
def check_goal_reached() -> str:
    """Check whether the robot has arrived at the destination.

    Returns a string starting with ``GOAL REACHED`` when within 0.5 m of
    the destination, otherwise ``NOT_REACHED`` plus the remaining distance.
    """
    ctrl = get_controller()
    x, y, _ = ctrl.get_pose()
    dist = math.hypot(ctrl.dest_x - x, ctrl.dest_y - y)
    if dist < 0.5:
        return (
            f"GOAL REACHED - {dist:.2f} meters from goal. "
            "Navigation complete!"
        )
    return (
        f"NOT_REACHED - {dist:.2f} meters remaining to destination "
        f"({ctrl.dest_x:.2f}, {ctrl.dest_y:.2f})."
    )


ALL_TOOLS = [
    move_forward,
    rotate,
    get_map_view,
    get_robot_pose,
    check_goal_reached,
]


__all__ = [
    "ALL_TOOLS",
    "MAP_TOOL_TEXT",
    "check_goal_reached",
    "get_map_view",
    "get_robot_pose",
    "move_forward",
    "rotate",
]
