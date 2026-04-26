"""Helpers for saving the exact multimodal LLM request payload."""

import base64
import json
from pathlib import Path
from typing import Any


class LlmRequestLogger:
    """Persist the exact text and image payload sent to the LLM."""

    def __init__(self, run_dir: Path):
        self._run_dir = run_dir

    def log_request(
        self,
        *,
        prefix: str,
        system_prompt: str,
        user_prompt: str,
        planning_context: dict[str, Any],
        reason: str,
        img_b64: str | None,
    ) -> None:
        request_dir = self._run_dir / "requests"
        request_dir.mkdir(parents=True, exist_ok=True)

        meta_path = request_dir / f"{prefix}_request.json"
        meta_path.write_text(
            json.dumps(
                {
                    "reason": reason,
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                    "planning_context": planning_context,
                    "has_image": img_b64 is not None,
                },
                indent=2,
            )
        )

        prompt_path = request_dir / f"{prefix}_user_prompt.txt"
        prompt_path.write_text(user_prompt)

        system_path = request_dir / f"{prefix}_system_prompt.txt"
        system_path.write_text(system_prompt)

        if img_b64 is not None:
            png_path = request_dir / f"{prefix}_request_image.png"
            png_path.write_bytes(base64.b64decode(img_b64))
