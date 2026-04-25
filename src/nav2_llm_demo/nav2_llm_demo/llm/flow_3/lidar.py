"""LiDAR sector summarization for flow_3.

Takes a ``LaserScan``-as-dict (the shape produced by ``llm_agent_node._scan_cb``)
and returns a compact, LLM-friendly text block. The LLM uses this alongside
the rasterized map image to decide whether ``move_forward`` / ``rotate`` is
safe.

Sectors (compass-style, 6 wedges of 60 degrees each, centered on cardinal
directions, with 0 deg = straight ahead and positive = counter-clockwise to
match the standard ROS REP-103 convention):

    front       -30 .. +30
    front_left  +30 .. +90
    back_left   +90 .. +150
    back       +150 .. +180  AND  -180 .. -150  (wraps)
    back_right -150 .. -90
    front_right -90 .. -30

Per-sector status (tuned for TurtleBot3 Burger, ~0.15 m radius):

    < 0.40 m         BLOCKED
    0.40 - 1.00 m    CAUTION
    >= 1.00 m        CLEAR
    no valid reading CLEAR (sensor reports inf -> nothing detected)
"""

from __future__ import annotations

import math
from typing import Any


# Sector boundaries in radians. (lo, hi]; ``lo`` may be greater than ``hi``
# when the sector wraps across +/- pi (the ``back`` sector does this).
_SECTORS: list[tuple[str, float, float]] = [
    ("front",       math.radians(-30),  math.radians(30)),
    ("front_left",  math.radians(30),   math.radians(90)),
    ("back_left",   math.radians(90),   math.radians(150)),
    ("back",        math.radians(150),  math.radians(-150)),  # wraps
    ("back_right",  math.radians(-150), math.radians(-90)),
    ("front_right", math.radians(-90),  math.radians(-30)),
]

BLOCKED_M = 0.40
CAUTION_M = 1.00


def _wrap_pi(angle: float) -> float:
    """Normalize an angle to (-pi, pi]."""
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle <= -math.pi:
        angle += 2.0 * math.pi
    return angle


def _in_sector(angle: float, lo: float, hi: float) -> bool:
    """True if ``angle`` (radians, normalized) lies in the (lo, hi] sector.

    Handles the wrap-around case where ``lo > hi`` (the ``back`` sector
    spans pi and -pi).
    """
    if lo <= hi:
        return lo < angle <= hi
    # wraps across +/- pi
    return angle > lo or angle <= hi


def _classify(distance_m: float) -> str:
    if distance_m < BLOCKED_M:
        return "BLOCKED"
    if distance_m < CAUTION_M:
        return "CAUTION"
    return "CLEAR"


def summarize_scan(scan: dict[str, Any]) -> str:
    """Render the scan dict as the multi-line text the LLM sees.

    Input shape matches what ``LlmAgentNode._scan_cb`` produces; missing
    fields fall back to safe defaults.
    """
    ranges = scan.get("ranges") or []
    angle_min = float(scan.get("angle_min", -math.pi))
    angle_inc = float(scan.get("angle_increment", 0.0)) or 0.0
    range_min = float(scan.get("range_min", 0.0))
    range_max = float(scan.get("range_max", math.inf))
    frame_id = scan.get("frame_id", "?")
    stamp_sec = float(scan.get("stamp_sec", 0.0))

    if not ranges or angle_inc <= 0.0:
        return (
            "LiDAR: no usable scan data "
            f"(ranges={len(ranges)}, angle_increment={angle_inc})."
        )

    # Per-sector running minimum (in meters). Use +inf so an empty sector
    # still gets classified CLEAR.
    sector_min: dict[str, float] = {name: math.inf for name, _, _ in _SECTORS}

    for i, raw in enumerate(ranges):
        try:
            r = float(raw)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(r):
            continue
        if r < range_min or r > range_max:
            continue
        angle = _wrap_pi(angle_min + i * angle_inc)
        for name, lo, hi in _SECTORS:
            if _in_sector(angle, lo, hi):
                if r < sector_min[name]:
                    sector_min[name] = r
                break

    lines: list[str] = []
    lines.append(
        f"LiDAR (frame={frame_id}, stamp={stamp_sec:.2f}s, "
        f"thresholds: BLOCKED < {BLOCKED_M:.2f} m, "
        f"CAUTION < {CAUTION_M:.2f} m):"
    )

    blocked: list[str] = []
    caution: list[str] = []
    for name, _, _ in _SECTORS:
        d = sector_min[name]
        if math.isinf(d):
            label = "CLEAR"
            d_str = "  inf  "
        else:
            label = _classify(d)
            d_str = f"{d:5.2f} m"
        marker = ""
        if label == "BLOCKED":
            marker = "   <-- do NOT move into this sector"
            blocked.append(name)
        elif label == "CAUTION":
            caution.append(name)
        lines.append(f"  - {name:<12}: {d_str}  {label}{marker}")

    front_d = sector_min["front"]
    if math.isinf(front_d):
        front_status = "CLEAR (no return)"
    else:
        front_status = f"{_classify(front_d)} ({front_d:.2f} m)"

    headline_bits = [f"forward is {front_status}"]
    if blocked:
        headline_bits.append("avoid " + ", ".join(blocked))
    if caution and not blocked:
        headline_bits.append("caution in " + ", ".join(caution))
    lines.append("Headline: " + "; ".join(headline_bits) + ".")

    return "\n".join(lines)


__all__ = ["summarize_scan", "BLOCKED_M", "CAUTION_M"]
