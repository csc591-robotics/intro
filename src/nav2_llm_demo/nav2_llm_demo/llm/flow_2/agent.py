"""Flow 2 agent: LangGraph prebuilt ``create_react_agent``.

The graph alternates an LLM node with a ``ToolNode`` until the LLM stops
emitting tool calls (or recursion limit fires). Because ``get_map_view``
returns a multimodal content list, the map image is delivered to the model
inside the ``ToolMessage`` itself; the agent loop never injects an extra
``HumanMessage`` for the image.

Public surface mirrors flow 1's ``VisionNavigationAgent`` so the ROS node
needs no per-flow conditionals:

* ``initialize(source_x, source_y, dest_x, dest_y)``
* ``step() -> str``
* ``goal_reached_in_last_step`` (property)
* ``run_dir`` (property)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage

try:  # langgraph >= 0.2.x
    from langgraph.errors import GraphRecursionError
except ImportError:  # pragma: no cover - older langgraph
    GraphRecursionError = RuntimeError  # type: ignore[assignment]

from langgraph.prebuilt import create_react_agent

from ..controller import make_run_dir, resolve_llm_config
from ..message_utils import prune_old_images
from .logging import PerCallLogger
from .prompt import SYSTEM_PROMPT
from .tools import ALL_TOOLS


class Flow2Agent:
    """ReAct-based vision navigation agent."""

    def __init__(
        self,
        provider: str | None = None,
        model_name: str | None = None,
        recursion_limit: int = 100,
    ) -> None:
        if provider is None or model_name is None:
            resolved_provider, resolved_model = resolve_llm_config("2")
            provider = provider or resolved_provider
            model_name = model_name or resolved_model

        self._provider = provider
        self._model_name = model_name

        self._llm = init_chat_model(
            model=model_name,
            model_provider=provider,
            temperature=0.1,
        )
        self._graph = create_react_agent(
            self._llm,
            tools=ALL_TOOLS,
            prompt=SYSTEM_PROMPT,
        )
        self._recursion_limit = int(recursion_limit)
        self._messages: list[Any] = []
        self._run_dir: Path | None = None
        self._logger: PerCallLogger | None = None
        self._completed = False
        self._final_summary: str | None = None

    # ------------------------------------------------------------------
    # Run / log directory
    # ------------------------------------------------------------------

    def _ensure_run_dir(self) -> Path:
        if self._run_dir is None:
            self._run_dir = make_run_dir(flow="2")
            self._logger = PerCallLogger(
                self._run_dir,
                provider=self._provider,
                model=self._model_name,
            )
        return self._run_dir

    # ------------------------------------------------------------------
    # Conversation setup
    # ------------------------------------------------------------------

    def initialize(
        self,
        source_x: float,
        source_y: float,
        dest_x: float,
        dest_y: float,
    ) -> None:
        """Set up the initial conversation.

        ``create_react_agent`` already prepends the SystemMessage built from
        ``SYSTEM_PROMPT``, so the initial state we hand it only needs the
        first HumanMessage describing the navigation goal.
        """
        self._ensure_run_dir()
        self._messages = [
            HumanMessage(content=(
                f"Navigate the robot to the destination.\n"
                f"Source (start): ({source_x:.2f}, {source_y:.2f})\n"
                f"Destination (goal): ({dest_x:.2f}, {dest_y:.2f})\n\n"
                "Begin by calling get_map_view() to see the map."
            )),
        ]
        self._completed = False
        self._final_summary = None

    # ------------------------------------------------------------------
    # Graph execution
    # ------------------------------------------------------------------

    def step(self) -> str:
        """Run the full ReAct graph once. Subsequent calls are no-ops.

        ``create_react_agent`` runs LLM/tool nodes in a loop internally
        until the LLM stops emitting tool calls (or the recursion limit
        fires). One ``step()`` call therefore corresponds to one complete
        navigation episode from the agent's perspective.

        Returns a short human-readable summary of what happened. The full
        per-LLM-call request/response artifacts are written by the
        ``PerCallLogger`` callback into ``llm_controls_call_NNN/`` folders
        under ``run_dir``.
        """
        if self._completed:
            return self._final_summary or "Agent already completed."

        self._ensure_run_dir()
        config: dict[str, Any] = {
            "recursion_limit": self._recursion_limit,
            "callbacks": [self._logger] if self._logger else [],
        }

        try:
            # Drop stale base64 images from history before sending. Anthropic
            # vision input bills each PNG ~1500-2000 tokens; without pruning,
            # ~30 turns blows past the 30k-tokens-per-minute dev tier limit.
            pruned = prune_old_images(self._messages, keep_last=1)
            result = self._graph.invoke(
                {"messages": pruned},
                config=config,
            )
            self._messages = list(result.get("messages", self._messages))
            summary = self._summarize(self._messages)
        except GraphRecursionError as exc:
            summary = (
                "Graph hit recursion limit "
                f"({self._recursion_limit}); agent stopped: {exc}"
            )
        except Exception as exc:  # noqa: BLE001 - surface any LLM error
            summary = f"Graph failed: {type(exc).__name__}: {exc}"

        self._completed = True
        self._final_summary = summary
        return summary

    @staticmethod
    def _summarize(messages: list[Any]) -> str:
        """Build a one-liner describing the most recent assistant action."""
        for msg in reversed(messages):
            tool_calls = getattr(msg, "tool_calls", None) or []
            if tool_calls:
                names = [tc.get("name", "?") for tc in tool_calls]
                return "Final tool calls: " + ", ".join(names)
            content = getattr(msg, "content", "")
            if isinstance(content, str) and content.strip():
                return f"Final assistant text: {content[:200]}"
        return "Agent completed with no recorded final message."

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def terminated(self) -> bool:
        """True once the graph has finished (success or unrecoverable error).

        The ROS-side loop polls this and bails out so we don't tight-loop
        on a cached failure summary (e.g. an Anthropic 429 that the agent
        already treated as terminal).
        """
        return self._completed

    @property
    def goal_reached_in_last_step(self) -> bool:
        """True if any recent ToolMessage contains a GOAL REACHED marker."""
        for msg in reversed(self._messages[-10:]):
            content = getattr(msg, "content", "")
            if isinstance(content, str) and "GOAL REACHED" in content:
                return True
            if isinstance(content, list):
                for part in content:
                    if (isinstance(part, dict)
                            and isinstance(part.get("text"), str)
                            and "GOAL REACHED" in part["text"]):
                        return True
        return False

    @property
    def run_dir(self) -> str | None:
        return str(self._run_dir) if self._run_dir else None


def build_agent(
    provider: str | None = None,
    model_name: str | None = None,
    recursion_limit: int = 100,
) -> Flow2Agent:
    """Build and return a flow 2 ``Flow2Agent``."""
    return Flow2Agent(
        provider=provider,
        model_name=model_name,
        recursion_limit=recursion_limit,
    )
