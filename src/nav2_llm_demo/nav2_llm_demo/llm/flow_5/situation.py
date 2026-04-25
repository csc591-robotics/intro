"""Build the per-cycle situation HumanMessage for flow_5.

Produces a multimodal LangChain ``HumanMessage`` with two content
blocks:

1. A text block with the robot's pose, the current target waypoint, the
   bearing-from-heading required to face it, and a SUGGESTED action
   computed deterministically from the geometry (``rotate(X)`` if
   |bearing| > 15 deg, otherwise ``move_forward(Y)``).
2. The map image with the planned magenta path drawn on it, the cyan
   crosshair on the current target, the red robot dot+arrow, blue
   source, and green destination.

The LLM's job is just to copy the SUGGESTED action into a tool call.
"""

from __future__ import annotations

import math
from typing import Any

from langchain_core.messages import HumanMessage

from ..controller import get_controller
from .renderer import render_with_path


_HEADER = (
    "Path-follower situation.\n"
    "- The MAGENTA line on the map is the pre-planned path. It was "
    "computed by A* on the rasterized map and is guaranteed to stay "
    "in free (white) space; trust it implicitly.\n"
    "- The CYAN crosshair is your CURRENT TARGET waypoint. The system "
    "auto-advances to the next waypoint when you are within 0.5 m.\n"
    "- The RED dot + small arrow is you and your heading.\n"
    "- BLUE = start, GREEN = final destination.\n"
    "Reply with EXACTLY ONE tool call: move_forward or rotate."
)


def _wrap_180(deg: float) -> float:
    """Normalize an angle in degrees to [-180, +180]."""
    while deg > 180.0:
        deg -= 360.0
    while deg < -180.0:
        deg += 360.0
    return deg


def build_situation_message(
    planned_path: list[tuple[float, float]],
    target_idx: int,
    *,
    map_yaml_path: str,
    source_xy: tuple[float, float],
    dest_xy: tuple[float, float],
    rotate_threshold_deg: float = 15.0,
    max_step_m: float = 0.6,
) -> tuple[HumanMessage, dict[str, Any]]:
    """Return ``(human_message, artifact)``.

    ``artifact`` carries the raw image + suggestion for logging /
    debugging; the agent does not feed it back to the LLM.
    """
    ctrl = get_controller()
    x, y, yaw = ctrl.get_pose()

    safe_idx = max(0, min(target_idx, len(planned_path) - 1))
    tx, ty = planned_path[safe_idx]
    dx, dy = tx - x, ty - y
    distance = math.hypot(dx, dy)
    bearing_world_deg = math.degrees(math.atan2(dy, dx))
    rotate_hint = _wrap_180(bearing_world_deg - math.degrees(yaw))

    if abs(rotate_hint) > rotate_threshold_deg:
        suggestion = f"rotate({rotate_hint:+.0f})"
    else:
        step = min(max_step_m, max(0.15, distance))
        suggestion = f"move_forward({step:.2f})"

    final_x, final_y = dest_xy
    final_dist = math.hypot(final_x - x, final_y - y)
    waypoints_remaining = max(0, len(planned_path) - safe_idx - 1)

    text = (
        _HEADER
        + "\n\n"
        + f"- Pose: ({x:.2f}, {y:.2f}), heading {math.degrees(yaw):+.1f} deg.\n"
        + f"- Current target: ({tx:.2f}, {ty:.2f}); "
        + f"distance {distance:.2f} m; "
        + f"bearing-from-heading {rotate_hint:+.1f} deg.\n"
        + f"- Final destination: ({final_x:.2f}, {final_y:.2f}); "
        + f"distance {final_dist:.2f} m.\n"
        + f"- Waypoints remaining after this one: {waypoints_remaining}.\n"
        + f"- SUGGESTED ACTION: {suggestion}"
    )

    img_b64 = render_with_path(
        map_yaml_path=map_yaml_path,
        robot_x=x, robot_y=y, robot_yaw=yaw,
        dest_x=final_x, dest_y=final_y,
        source_x=source_xy[0], source_y=source_xy[1],
        planned_path=planned_path,
        target_idx=safe_idx,
        crop_radius_m=18.0,
        output_size=512,
    )

    msg = HumanMessage(content=[
        {"type": "text", "text": text},
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{img_b64}",
            },
        },
    ])

    artifact = {
        "image_b64": img_b64,
        "robot_pose": {"x": x, "y": y, "yaw": yaw},
        "target_idx": safe_idx,
        "target_xy": [tx, ty],
        "rotate_hint_deg": rotate_hint,
        "distance_to_target_m": distance,
        "distance_to_goal_m": final_dist,
        "suggestion": suggestion,
    }
    return msg, artifact


__all__ = ["build_situation_message"]
