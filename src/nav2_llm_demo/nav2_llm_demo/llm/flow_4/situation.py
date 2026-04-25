"""Build the per-cycle situation HumanMessage for flow_4.

Mirrors what flow_3's ``get_situation`` tool returns, but exposed as a
plain Python function so the flow_4 graph can call it directly without
going through the LangChain tool layer (since the LLM in flow_4 doesn't
have ``get_situation`` in its tool inventory).
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage

from ..controller import get_controller
from ..map_renderer import render_annotated_map
from ..flow_3.interpreter import interpret_lidar
from ..flow_3.lidar import summarize_scan


_HEADER = (
    "SITUATION (live map + LiDAR safety read).\n"
    "1) Image below: top-down map. Legend:\n"
    "   - BLUE circle  = where you started.\n"
    "   - RED dot      = your current position.\n"
    "   - The small RED arrow on the dot points where the robot is FACING.\n"
    "     'Forward' = direction of the arrow tip.\n"
    "     rotate(positive) tips arrow CCW; rotate(negative) tips arrow CW.\n"
    "   - GREEN plus / circle = your destination.\n"
    "   - WHITE = free; BLACK = forbidden obstacles; GRAY = unknown.\n"
    "2) Below the map is the LiDAR analyst read. Treat the FRONT distance\n"
    "   as the most important number for whether move_forward is safe.\n"
    "3) Reply with EXACTLY ONE tool call: move_forward or rotate."
)


def build_situation_message() -> tuple[HumanMessage, dict]:
    """Snapshot the world and return ``(human_message, artifact_dict)``.

    The HumanMessage is what gets appended to the conversation. The
    artifact dict mirrors what flow_3's tool returns and is consumed by
    flow_4's logger to dump the raw PNG / lidar separately.
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
            "LiDAR not available yet (no /scan received). Be conservative."
        )
    else:
        sector_text = summarize_scan(raw_scan)
        interpreted = interpret_lidar(sector_text)

    text_block = (
        _HEADER
        + "\n\nLiDAR analyst read:\n"
        + interpreted
    )

    msg = HumanMessage(content=[
        {"type": "text", "text": text_block},
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{img_b64}",
            },
        },
    ])

    artifact = {
        "image_b64": img_b64,
        "raw_lidar_summary": sector_text,
        "interpreted_lidar": interpreted,
        "robot_pose": {"x": x, "y": y, "yaw": yaw},
    }
    return msg, artifact


__all__ = ["build_situation_message"]
