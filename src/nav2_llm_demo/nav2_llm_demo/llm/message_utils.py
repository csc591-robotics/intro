"""Message-history utilities shared by flow_2 and flow_3.

The vision-LLM ReAct loop appends a fresh map image to the conversation
every time the LLM calls ``get_map_view``. After a few dozen turns the
history embeds many base64-encoded PNGs, each of which Anthropic Claude
counts as ~1500-2000 input tokens for vision pricing. The dev-tier rate
limit (30k input tokens / minute) is then trivial to blow past.

``prune_old_images`` walks a LangChain message list and replaces every
``image_url`` content block EXCEPT the most recent one(s) with a small
text placeholder. The conversation flow stays intact (tool_call_ids,
text content, ordering); only stale image bytes are dropped.
"""

from __future__ import annotations

from copy import copy
from typing import Any


_PRUNED_PLACEHOLDER = (
    "[earlier map image pruned to save tokens; only the most recent map "
    "view is kept verbatim. Call get_map_view() again if you need a "
    "fresh image]"
)


def _content_has_image(content: Any) -> bool:
    if not isinstance(content, list):
        return False
    for part in content:
        if isinstance(part, dict) and part.get("type") == "image_url":
            return True
    return False


def _strip_images(content: list[Any]) -> list[Any]:
    """Return a new content list with image_url blocks replaced by a single
    text placeholder. Other text / non-image blocks are preserved."""
    out: list[Any] = []
    placeholder_emitted = False
    for part in content:
        if (isinstance(part, dict)
                and part.get("type") == "image_url"):
            if not placeholder_emitted:
                out.append({"type": "text", "text": _PRUNED_PLACEHOLDER})
                placeholder_emitted = True
            continue
        out.append(part)
    if not out:
        out.append({"type": "text", "text": _PRUNED_PLACEHOLDER})
    return out


def prune_old_images(messages: list[Any], keep_last: int = 1) -> list[Any]:
    """Return a fresh list with all but the most recent ``keep_last`` images
    replaced by text placeholders.

    The original message objects are NOT mutated; we shallow-copy each
    affected message (via ``copy.copy``) and rewrite ``content`` on the
    copy. Messages without inline images pass through unchanged.

    ``keep_last`` defaults to 1 (only the latest image survives). Set to
    a larger value if you want a small rolling window of recent maps.
    """
    if keep_last < 0:
        keep_last = 0

    image_positions = [
        i for i, m in enumerate(messages)
        if _content_has_image(getattr(m, "content", None))
    ]

    if len(image_positions) <= keep_last:
        return list(messages)

    cutoff = (
        image_positions[-keep_last] if keep_last > 0
        else len(messages)
    )

    pruned: list[Any] = []
    for i, msg in enumerate(messages):
        content = getattr(msg, "content", None)
        if i < cutoff and _content_has_image(content):
            new_msg = copy(msg)
            new_msg.content = _strip_images(content)  # type: ignore[attr-defined]
            pruned.append(new_msg)
        else:
            pruned.append(msg)
    return pruned


__all__ = ["prune_old_images"]
