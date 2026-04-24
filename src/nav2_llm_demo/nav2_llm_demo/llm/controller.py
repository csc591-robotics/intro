"""Shared types and helpers used by every LLM flow.

Each flow (flow_1, flow_2, ...) implements its own agent loop, but they all
need the same things:

- A ``RobotController`` Protocol that the ROS node implements (move_forward,
  rotate, get_pose, plus map/source/destination metadata used by the map
  renderer).
- A module-level controller registry so tools can call back into the ROS
  node from any thread/coroutine without passing it around explicitly.
- A consistent on-disk layout for per-run debug artifacts:

      <WORKSPACE_DIR>/llm_agent_runs/flow_<LLM_FLOW>/<timestamp>/

  ``WORKSPACE_DIR`` defaults to ``/workspace`` (the bind-mount point inside
  this project's Docker container), which maps to the host's ``intro/``
  folder. ``LLM_FLOW`` defaults to ``1``.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Protocol


# ---------------------------------------------------------------------------
# Robot controller protocol
# ---------------------------------------------------------------------------

class RobotController(Protocol):
    """Interface the agent tools call into.  Implemented by the ROS node."""

    def get_pose(self) -> tuple[float, float, float]:
        ...

    def move_forward(self, distance_m: float, speed: float = 0.15) -> str:
        ...

    def rotate(self, angle_deg: float, speed: float = 0.5) -> str:
        ...

    @property
    def map_yaml_path(self) -> str: ...
    @property
    def source_x(self) -> float: ...
    @property
    def source_y(self) -> float: ...
    @property
    def dest_x(self) -> float: ...
    @property
    def dest_y(self) -> float: ...


# ---------------------------------------------------------------------------
# Module-level controller registry
# ---------------------------------------------------------------------------

_controller: RobotController | None = None


def set_controller(ctrl: RobotController) -> None:
    """Register the active RobotController.  Called once by the ROS node."""
    global _controller
    _controller = ctrl


def get_controller() -> RobotController:
    """Return the registered controller; raise if it hasn't been set."""
    if _controller is None:
        raise RuntimeError(
            "RobotController not registered; call set_controller first"
        )
    return _controller


# ---------------------------------------------------------------------------
# Per-run output directory
# ---------------------------------------------------------------------------

def make_run_dir(flow: str | int | None = None) -> Path:
    """Create and return ``<workspace>/llm_agent_runs/flow_<N>/<timestamp>/``.

    ``flow`` defaults to ``$LLM_FLOW`` (which itself defaults to ``"1"``).
    Each call generates a fresh timestamp so two agents launched in the same
    second still get separate folders.
    """
    if flow is None:
        flow = os.environ.get("LLM_FLOW", "1").strip() or "1"
    workspace = os.environ.get("WORKSPACE_DIR", "/workspace")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(workspace) / "llm_agent_runs" / f"flow_{flow}" / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


# ---------------------------------------------------------------------------
# Per-flow LLM provider / model resolution
# ---------------------------------------------------------------------------

def resolve_llm_config(flow: str | int) -> tuple[str, str]:
    """Pick the LLM provider and model for a given flow.

    Lookup order:

    1. ``FLOW{N}_LLM_PROVIDER`` / ``FLOW{N}_LLM_MODEL`` (per-flow override).
    2. ``LLM_PROVIDER`` / ``LLM_MODEL`` (global fallback).

    Raises ``RuntimeError`` if neither is set, with a message that points to
    the exact env vars to add.
    """
    n = str(flow).strip() or "1"

    provider = (
        os.environ.get(f"FLOW{n}_LLM_PROVIDER", "").strip()
        or os.environ.get("LLM_PROVIDER", "").strip()
    )
    model = (
        os.environ.get(f"FLOW{n}_LLM_MODEL", "").strip()
        or os.environ.get("LLM_MODEL", "").strip()
    )

    if not provider:
        raise RuntimeError(
            f"No LLM provider for flow {n}. Set FLOW{n}_LLM_PROVIDER "
            "(per-flow) or LLM_PROVIDER (global) in your .env."
        )
    if not model:
        raise RuntimeError(
            f"No LLM model for flow {n}. Set FLOW{n}_LLM_MODEL (per-flow) "
            "or LLM_MODEL (global) in your .env."
        )
    return provider, model
