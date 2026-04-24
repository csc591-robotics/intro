"""Flow 1: hand-rolled custom agent loop with multimodal HumanMessage injection.

This is the original implementation. Each ``get_map_view`` tool call returns
a short text result, and the loop appends a separate ``HumanMessage``
carrying the actual map image as a multimodal content block. Movement tools
do *not* trigger an automatic image refresh; the LLM has to call
``get_map_view`` explicitly to see a new map.
"""

from .agent import VisionNavigationAgent, build_agent

__all__ = ["VisionNavigationAgent", "build_agent"]
