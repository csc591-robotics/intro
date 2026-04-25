"""Message-history utilities shared by flow_2 and flow_3.

Two complementary trimmers:

* ``prune_old_images`` -- replaces every ``image_url`` content block
  except the most recent one(s) with a small text placeholder. Preserves
  the surrounding conversation structure entirely.
* ``compact_history`` -- drops most of the middle of the conversation,
  keeping only the very first message (the original goal HumanMessage)
  and the last K "rounds" (an AIMessage with tool_calls plus its
  matching ToolMessage(s)). Optionally injects a HumanMessage reminder
  in the gap so the LLM is re-shown the key rules every turn even when
  earlier turns are gone.

Why both? Image pruning alone keeps the message *count* the same, which
still costs many tokens once the conversation is long. ``compact_history``
removes whole turns so the request stays small even after many minutes
of navigation. Apply ``compact_history`` first, then ``prune_old_images``.
"""

from __future__ import annotations

from copy import copy
from typing import Any

from langchain_core.messages import HumanMessage


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


def _is_action_msg(msg: Any) -> bool:
    """True iff this is an AIMessage carrying tool_calls (start of a round)."""
    tcs = getattr(msg, "tool_calls", None) or []
    return bool(tcs)


def compact_history(
    messages: list[Any],
    keep_rounds: int = 4,
    reminder_text: str | None = None,
) -> list[Any]:
    """Trim the middle of a long ReAct conversation.

    Output layout (when compaction triggers):

        [original first message]                  <-- preserves goal context
        [HumanMessage(reminder_text)]             <-- only if reminder_text given
        [last `keep_rounds` AIMessage+ToolMessage chunks]

    The cut is placed *just before* the (keep_rounds)-th-from-last
    AIMessage with tool_calls, which guarantees that every kept
    ``ToolMessage`` still has its parent ``AIMessage`` in the kept set
    (Anthropic rejects orphan tool_use_ids).

    If the conversation is short enough that compaction would either be
    trivial (cut at index 0) or skip nothing useful, the original list
    is returned unchanged.
    """
    if keep_rounds < 1:
        keep_rounds = 1

    if not messages:
        return list(messages)

    action_indices = [i for i, m in enumerate(messages) if _is_action_msg(m)]

    if len(action_indices) <= keep_rounds:
        # Not enough action rounds to bother compacting.
        return list(messages)

    cut = action_indices[-keep_rounds]

    if cut <= 1:
        # Cutting only the initial HumanMessage isn't useful.
        return list(messages)

    head = list(messages[:1])
    middle: list[Any] = []
    if reminder_text:
        middle.append(HumanMessage(content=reminder_text))
    tail = list(messages[cut:])

    return head + middle + tail


__all__ = ["prune_old_images", "compact_history"]
