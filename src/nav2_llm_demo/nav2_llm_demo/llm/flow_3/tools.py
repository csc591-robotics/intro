"""Tool definitions for flow_3 (ReAct + map image + LiDAR summary).

Identical to flow_2's tools except for one addition: ``get_lidar_summary``
returns a 6-sector classification (BLOCKED / CAUTION / CLEAR) of the live
``/scan`` topic, so the LLM can cross-reference the rasterized map against
what the robot actually sees right now.

The flow_3 system prompt requires the LLM to call BOTH ``get_map_view`` and
``get_lidar_summary`` before every ``move_forward`` / ``rotate``.
"""

from __future__ import annotations

import json
import math

from langchain_core.tools import tool

from ..controller import get_controller
from ..map_renderer import render_annotated_map
from .interpreter import interpret_lidar
from .lidar import summarize_scan


# Reiterated on every map view so the legend stays adjacent to the image
# even after many turns dilute the system prompt.
MAP_TOOL_TEXT = (
    "Current map view. Legend:\n"
    "- BLUE circle  = where I started.\n"
    "- RED dot      = where I am right now. The small RED arrow on the dot "
    "is my heading (the direction I am facing).\n"
    "- GREEN plus / circle = my destination.\n"
    "- WHITE pixels = free space; I am ALLOWED to drive here.\n"
    "- BLACK pixels = walls / obstacles per the rasterized map. The map can "
    "be wrong (missing or extra obstacles), so always cross-check with "
    "get_lidar_summary before moving.\n"
    "- GRAY pixels  = unknown; treat as unsafe.\n"
    "Reminder: before issuing move_forward or rotate, call "
    "get_lidar_summary() to verify the chosen direction is not BLOCKED. "
    "If the map and LiDAR disagree, trust LiDAR for short-range safety."
)


@tool
def move_forward(distance_meters: float) -> str:
    """Drive the robot forward (positive) or backward (negative) by the given
    number of meters. Use small steps (0.3 to 1.0 m). Robot stops on its
    own when the distance is reached. Only call this after the most recent
    map view AND lidar summary both indicate the direction is safe.
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
def get_lidar_summary() -> str:
    """Return a 6-sector summary of the latest LiDAR scan.

    Each sector reports the closest obstacle distance and a status label:
    BLOCKED (< 0.40 m, do not enter), CAUTION (< 1.00 m, stay alert), or
    CLEAR. Use this when you specifically want raw sector numbers.
    Otherwise prefer get_situation(), which returns the map AND a
    plain-English LiDAR interpretation in one tool call.
    """
    ctrl = get_controller()
    raw = ctrl.get_latest_scan() if hasattr(ctrl, "get_latest_scan") else None
    if raw is None:
        return (
            "LiDAR not available yet (no /scan message received). Try "
            "again in a moment, or proceed cautiously based on the map alone."
        )
    return summarize_scan(raw)


SITUATION_HEADER = (
    "SITUATION (live map + LiDAR safety read).\n"
    "1) The image below is a fresh top-down map. Same legend as before:\n"
    "   - BLUE circle  = where you started.\n"
    "   - RED dot      = your current position.\n"
    "   - The small RED arrow on the dot points where the robot is FACING.\n"
    "     'Forward' is the direction the arrow points to.\n"
    "     'Rotate left' (positive degrees) tips the arrow CCW on the image.\n"
    "     'Rotate right' (negative degrees) tips the arrow CW on the image.\n"
    "   - GREEN plus / circle = your destination.\n"
    "   - WHITE pixels = free space, OK to drive on.\n"
    "   - BLACK pixels = obstacles. NEVER drive into black; this is\n"
    "     non-negotiable. If your RED dot is currently ON or touching a\n"
    "     BLACK region, your IMMEDIATE next action MUST be rotate(180)\n"
    "     followed by move_forward to escape.\n"
    "   - GRAY pixels  = unknown; treat as unsafe.\n"
    "2) Below the map is the LiDAR safety read (interpreted by an analyst\n"
    "   model). Treat the FRONT distance as the most important number;\n"
    "   it tells you whether move_forward is safe RIGHT NOW.\n"
    "3) LiDAR is a SAFETY REINFORCEMENT only. The map is the authority\n"
    "   for 'where can I go'. LiDAR catches what the map missed (dynamic\n"
    "   objects, mis-rasterized walls); it cannot grant permission to\n"
    "   enter a BLACK region of the map."
)


@tool(response_format="content_and_artifact")
def get_situation() -> tuple[list[dict], dict]:
    """PREFERRED situational-awareness tool. Returns BOTH a fresh map
    image and a plain-English LiDAR interpretation in a single tool
    result, saving an extra round-trip. Call this before any
    move_forward / rotate decision.
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

    raw_scan = (
        ctrl.get_latest_scan()
        if hasattr(ctrl, "get_latest_scan")
        else None
    )
    if raw_scan is None:
        sector_text = ""
        interpreted = (
            "LiDAR not available yet (no /scan received). Use the map "
            "alone, but be conservative."
        )
    else:
        sector_text = summarize_scan(raw_scan)
        interpreted = interpret_lidar(sector_text)

    text_block = (
        SITUATION_HEADER
        + "\n\nLiDAR analyst read:\n"
        + interpreted
        + "\n\nNow look at the map image and pick the next move."
    )

    content: list[dict] = [
        {"type": "text", "text": text_block},
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{img_b64}",
            },
        },
    ]
    artifact = {
        "image_b64": img_b64,
        "raw_lidar_summary": sector_text,
        "interpreted_lidar": interpreted,
    }
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


_GOAL_THRESHOLD_M = 0.3  # ~robot footprint; matches flow_4 / flow_5 / node default


@tool
def check_goal_reached() -> str:
    """Check whether the robot has arrived at the destination.

    Returns a string starting with ``GOAL REACHED`` when within
    ``_GOAL_THRESHOLD_M`` meters of the destination, otherwise
    ``NOT_REACHED`` plus the remaining distance.
    """
    ctrl = get_controller()
    x, y, _ = ctrl.get_pose()
    dist = math.hypot(ctrl.dest_x - x, ctrl.dest_y - y)
    if dist < _GOAL_THRESHOLD_M:
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
    get_situation,
    get_map_view,
    get_lidar_summary,
    get_robot_pose,
    check_goal_reached,
]


__all__ = [
    "ALL_TOOLS",
    "MAP_TOOL_TEXT",
    "SITUATION_HEADER",
    "check_goal_reached",
    "get_lidar_summary",
    "get_map_view",
    "get_robot_pose",
    "get_situation",
    "move_forward",
    "rotate",
]
