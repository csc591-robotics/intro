"""Flow_6: pure Nav2 navigator (no LLM in the loop).

Drops into the same ``initialize / step / goal_reached_in_last_step /
terminated / run_dir`` interface as the LLM flows so the existing ROS
node and the ``nav2_llm_experiments`` orchestrator drive it without any
per-flow conditionals. Internally, ``Flow6Agent`` sends a single
``NavigateToPose`` action goal at ``initialize()`` time and the rest of
the loop just polls its status.
"""

from .agent import Flow6Agent, build_agent

__all__ = ["Flow6Agent", "build_agent"]
