"""LLM helpers for vision-based agent navigation.

Three flows are available; the active one is picked by the ``LLM_FLOW``
environment variable (default ``"1"``):

* ``LLM_FLOW=1`` -> :mod:`nav2_llm_demo.llm.flow_1` -- the original custom
  loop with multimodal HumanMessage injection after ``get_map_view``.
* ``LLM_FLOW=2`` -> :mod:`nav2_llm_demo.llm.flow_2` -- LangGraph
  ``create_react_agent``; ``get_map_view`` returns the map image inside
  the ``ToolMessage`` itself (no extra HumanMessage).
* ``LLM_FLOW=3`` -> :mod:`nav2_llm_demo.llm.flow_3` -- same as flow_2,
  but adds a ``get_lidar_summary`` tool. The system prompt forces the
  LLM to call BOTH ``get_map_view`` and ``get_lidar_summary`` before any
  ``move_forward`` / ``rotate``, so it can reconcile the rasterized map
  against the live ``/scan``.

All flows expose a ``build_agent()`` factory that returns an object with
the same minimal surface (``initialize``, ``step``,
``goal_reached_in_last_step``, ``run_dir``) so the ROS node never needs
per-flow conditionals.
"""

from __future__ import annotations

import os

from .controller import RobotController, get_controller, set_controller
from .map_renderer import render_annotated_map, render_full_map


_FLOW = os.environ.get("LLM_FLOW", "1").strip() or "1"

if _FLOW == "3":
    from .flow_3 import build_agent  # noqa: F401
elif _FLOW == "2":
    from .flow_2 import build_agent  # noqa: F401
elif _FLOW == "1":
    from .flow_1 import build_agent  # noqa: F401
else:
    raise RuntimeError(
        f"LLM_FLOW={_FLOW!r} is not supported. "
        "Set LLM_FLOW to 1, 2, or 3."
    )


__all__ = [
    "RobotController",
    "build_agent",
    "get_controller",
    "render_annotated_map",
    "render_full_map",
    "set_controller",
]
