"""LLM helpers for vision-based agent navigation.

Six flows are available; the active one is picked by the ``LLM_FLOW``
environment variable (default ``"1"``):

* ``LLM_FLOW=1`` -> :mod:`nav2_llm_demo.llm.flow_1` -- the original custom
  loop with multimodal HumanMessage injection after ``get_map_view``.
* ``LLM_FLOW=2`` -> :mod:`nav2_llm_demo.llm.flow_2` -- LangGraph
  ``create_react_agent``; ``get_map_view`` returns the map image inside
  the ``ToolMessage`` itself (no extra HumanMessage).
* ``LLM_FLOW=3`` -> :mod:`nav2_llm_demo.llm.flow_3` -- same as flow_2,
  plus a ``get_lidar_summary`` tool and a combined ``get_situation`` tool
  (image + LiDAR-analyst text in one ToolMessage). The LLM still chooses
  freely (ReAct freedom).
* ``LLM_FLOW=4`` -> :mod:`nav2_llm_demo.llm.flow_4` -- FIXED graph (no
  ReAct freedom). Each cycle: gather situation (Python; no LLM) -> decide
  (LLM with ONLY move_forward/rotate, tool_choice=any) -> execute ->
  check goal (< 1.0 m). The LLM cannot "look without acting".
* ``LLM_FLOW=5`` -> :mod:`nav2_llm_demo.llm.flow_5` -- A* path planner
  runs ONCE at init on the inflated PGM and produces a polyline of
  waypoints. The LLM follows the magenta line cycle by cycle with the
  bearing pre-computed for it ("rotate X" or "move_forward Y"). No
  LiDAR; no per-cycle planning. Most reliable on real maps.
* ``LLM_FLOW=6`` -> :mod:`nav2_llm_demo.llm.flow_6` -- pure Nav2
  navigator. No LLM at all: a ``NavigateToPose`` action client
  dispatches a single goal to the Nav2 BT navigator and reports
  ``distance_remaining`` until ``STATUS_SUCCEEDED`` /
  ``STATUS_ABORTED``. Used as a deterministic baseline.

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

if _FLOW == "6":
    from .flow_6 import build_agent  # noqa: F401
elif _FLOW == "5":
    from .flow_5 import build_agent  # noqa: F401
elif _FLOW == "4":
    from .flow_4 import build_agent  # noqa: F401
elif _FLOW == "3":
    from .flow_3 import build_agent  # noqa: F401
elif _FLOW == "2":
    from .flow_2 import build_agent  # noqa: F401
elif _FLOW == "1":
    from .flow_1 import build_agent  # noqa: F401
else:
    raise RuntimeError(
        f"LLM_FLOW={_FLOW!r} is not supported. "
        "Set LLM_FLOW to 1, 2, 3, 4, 5, or 6."
    )


__all__ = [
    "RobotController",
    "build_agent",
    "get_controller",
    "render_annotated_map",
    "render_full_map",
    "set_controller",
]
