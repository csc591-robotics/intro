"""Per-LLM-call logging for flow 2.

We attach a LangChain ``BaseCallbackHandler`` to ``create_react_agent``'s
LLM calls and write the same on-disk format flow 1 uses:

    <run_dir>/
        llm_controls_call_001/
            request.json
            response.json
            image_sent.png        (only when the request had an inline PNG)
        llm_controls_call_002/
        ...

No other files are written. The handler keeps a 1-based call counter and a
mapping of in-flight ``run_id`` -> dir so the start/end callbacks line up
correctly even if LangChain dispatches them on different threads.
"""

from __future__ import annotations

import base64
import json
import os
import threading
from pathlib import Path
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult


def _extract_last_image_b64(messages: list[Any]) -> str | None:
    """Find the most recent inline base64 PNG in any message's content."""
    for msg in reversed(messages):
        content = getattr(msg, "content", None)
        if not isinstance(content, list):
            continue
        for part in reversed(content):
            if not isinstance(part, dict):
                continue
            if part.get("type") != "image_url":
                continue
            url = (part.get("image_url") or {}).get("url", "")
            if "base64," in url:
                return url.split("base64,", 1)[1]
    return None


def _message_to_dict(msg: Any) -> dict[str, Any]:
    """Same shape as flow 1's ``_message_to_dict`` so request.json files
    from both flows can be diffed directly."""
    out: dict[str, Any] = {"role": type(msg).__name__}

    tool_calls = getattr(msg, "tool_calls", None)
    if tool_calls:
        out["tool_calls"] = tool_calls

    tool_call_id = getattr(msg, "tool_call_id", None)
    if tool_call_id:
        out["tool_call_id"] = tool_call_id

    content = getattr(msg, "content", None)
    if isinstance(content, str):
        out["content"] = content
    elif isinstance(content, list):
        parts: list[dict[str, Any]] = []
        for part in content:
            if not isinstance(part, dict):
                parts.append({"raw": str(part)[:500]})
                continue
            if part.get("type") == "image_url":
                parts.append({
                    "type": "image_url",
                    "image_url": {
                        "url": "<inline PNG, see image_sent.png>",
                    },
                })
            else:
                parts.append(part)
        out["content"] = parts
    elif content is not None:
        out["content"] = str(content)[:2000]

    return out


def _serialized_messages_to_objects(serialized_messages: list[list[dict]]) -> list[BaseMessage]:
    """Best-effort conversion of LangChain's serialized prompt format.

    Some providers pass the start callback ``messages: list[list[BaseMessage]]``
    while others pass dict-serialized variants. We just want to feed our
    helpers; if everything's already a BaseMessage we pass it through.
    """
    if not serialized_messages:
        return []
    flat: list[Any] = []
    for batch in serialized_messages:
        flat.extend(batch)
    return flat


class PerCallLogger(BaseCallbackHandler):
    """Write per-LLM-call request/response artifacts to ``run_dir``.

    Use one instance per agent run. ``run_dir`` is created lazily by the
    agent (see ``Flow2Agent``) so the callback can always count on it
    existing by the time the first chat-model start fires.
    """

    raise_error = False

    def __init__(
        self,
        run_dir: Path,
        provider: str = "",
        model: str = "",
    ) -> None:
        super().__init__()
        self._run_dir = Path(run_dir)
        self._provider = provider
        self._model = model
        self._call_num = 0
        self._lock = threading.Lock()
        self._dir_for_run: dict[UUID, Path] = {}
        self._num_for_run: dict[UUID, int] = {}

    @property
    def call_num(self) -> int:
        return self._call_num

    def _next_call_dir(self, run_id: UUID) -> tuple[int, Path]:
        with self._lock:
            self._call_num += 1
            n = self._call_num
            d = self._run_dir / f"llm_controls_call_{n:03d}"
            d.mkdir(parents=True, exist_ok=True)
            self._dir_for_run[run_id] = d
            self._num_for_run[run_id] = n
        return n, d

    def _dir_lookup(self, run_id: UUID) -> tuple[int | None, Path | None]:
        with self._lock:
            return self._num_for_run.get(run_id), self._dir_for_run.get(run_id)

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        flat_msgs = _serialized_messages_to_objects(messages)
        n, call_dir = self._next_call_dir(run_id)

        img_b64 = _extract_last_image_b64(flat_msgs)
        if img_b64:
            try:
                (call_dir / "image_sent.png").write_bytes(
                    base64.b64decode(img_b64)
                )
            except (ValueError, OSError):
                img_b64 = None

        request_payload = {
            "llm_call_num": n,
            "agent_step": n,
            "model": self._model
                or (serialized or {}).get("name")
                or os.environ.get("LLM_MODEL", ""),
            "provider": self._provider
                or os.environ.get("LLM_PROVIDER", ""),
            "image_sent": "image_sent.png" if img_b64 else None,
            "messages": [_message_to_dict(m) for m in flat_msgs],
        }
        try:
            (call_dir / "request.json").write_text(
                json.dumps(request_payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError:
            pass

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        n, call_dir = self._dir_lookup(run_id)
        if call_dir is None:
            return

        out: dict[str, Any] = {
            "llm_call_num": n,
            "role": "AIMessage",
            "content": "",
            "tool_calls": [],
        }
        try:
            generations = response.generations or []
            if generations and generations[0]:
                gen = generations[0][0]
                msg = getattr(gen, "message", None)
                if msg is not None:
                    out["role"] = type(msg).__name__
                    content = getattr(msg, "content", "")
                    if not isinstance(content, str):
                        content = str(content)
                    out["content"] = content
                    out["tool_calls"] = (
                        getattr(msg, "tool_calls", None) or []
                    )
                else:
                    out["content"] = getattr(gen, "text", "") or ""
        except Exception as exc:
            out["content"] = f"<error extracting response: {exc}>"

        try:
            (call_dir / "response.json").write_text(
                json.dumps(out, indent=2, ensure_ascii=False, default=str),
                encoding="utf-8",
            )
        except OSError:
            pass

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        n, call_dir = self._dir_lookup(run_id)
        if call_dir is None:
            return
        try:
            (call_dir / "response.json").write_text(
                json.dumps(
                    {
                        "llm_call_num": n,
                        "error": str(error),
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
        except OSError:
            pass


__all__ = ["PerCallLogger"]
