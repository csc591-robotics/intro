"""Flow_3: ``create_react_agent`` + map image + LiDAR.

Same loop / log format as :mod:`nav2_llm_demo.llm.flow_2` but with one
extra tool: ``get_lidar_summary``. The system prompt forces the LLM to
call BOTH ``get_map_view`` and ``get_lidar_summary`` before every
``move_forward`` / ``rotate``, so the LLM can reconcile the rasterized map
against the live ``/scan`` topic and avoid driving into PGM artifacts.
"""

from .agent import Flow3Agent, build_agent

__all__ = ["Flow3Agent", "build_agent"]
