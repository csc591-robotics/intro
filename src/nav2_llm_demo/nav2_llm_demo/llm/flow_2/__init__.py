"""Flow 2: LangGraph ``create_react_agent`` with multimodal ToolMessage.

Differences from flow 1:

- The agent loop is the prebuilt ``create_react_agent`` graph (LLM node
  alternating with a ToolNode).
- ``get_map_view`` returns a multimodal content list directly, so the map
  image is delivered to the model inside the ``ToolMessage`` itself instead
  of via a follow-up ``HumanMessage``.
- The system prompt strongly suggests calling ``get_map_view`` after every
  movement command, keeping the visual context fresh without us hardcoding
  any auto-injection logic in the loop.
"""

from .agent import Flow2Agent, build_agent

__all__ = ["Flow2Agent", "build_agent"]
