"""Flow_4: fixed-topology cycle (no ReAct freedom).

Each ``step()`` runs exactly one cycle:

    gather_situation (Python; no LLM)
        -> decide_action (LLM, tools = move_forward/rotate, tool_choice=any)
            -> execute_action (Python; runs the chosen tool)
                -> check_goal (Python; goal reached if dist < 1.0 m)

The LLM cannot "look without acting". Every cycle terminates in exactly
one ``move_forward`` or ``rotate`` call. Compare with flow_3 where the
LangGraph ``create_react_agent`` lets the model freely chain
``get_situation`` calls or dither between rotations.
"""

from .agent import Flow4Agent, build_agent

__all__ = ["Flow4Agent", "build_agent"]
