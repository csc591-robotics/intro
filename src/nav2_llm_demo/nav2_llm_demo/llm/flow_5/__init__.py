"""Flow_5: A* deterministic planning + LLM follower.

A* runs ONCE at init on the inflated PGM and produces a polyline of
waypoints from source to destination through guaranteed-free space.
Each cycle the LLM sees the map with that path drawn on it (magenta
polyline + cyan current-target crosshair) and a precomputed "rotate X
deg, move Y m" suggestion. The LLM emits exactly one move_forward or
rotate per cycle (tool_choice="any").

Compared to flow_1-4, this removes the part vision LLMs are worst at
(figuring out which way to go around obstacles) and keeps only the
visual servoing piece (rotate to face the cyan target, then move
forward). LiDAR is not used because A* already avoids obstacles.
"""

from .agent import Flow5Agent, build_agent

__all__ = ["Flow5Agent", "build_agent"]
