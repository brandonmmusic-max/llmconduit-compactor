"""When to compact + cheap token estimation.

Estimation is pluggable: exact via tiktoken if installed, else a calibrated
char/word heuristic (good enough for a trigger; we don't need exactness).
"""
from __future__ import annotations

from typing import List

try:  # optional exact tokenizer
    import tiktoken  # type: ignore
    _ENC = tiktoken.get_encoding("cl100k_base")
except Exception:  # pragma: no cover
    _ENC = None


def estimate_tokens(messages: List[dict]) -> int:
    parts: List[str] = []
    for m in messages:
        c = m.get("content")
        if isinstance(c, str):
            parts.append(c)
        elif isinstance(c, list):
            for block in c:
                if isinstance(block, dict):
                    parts.append(str(block.get("text", "")))
                    parts.append(str(block.get("content", "")))
        for tc in m.get("tool_calls", []) or []:
            parts.append(str(tc))
    text = "\n".join(parts)
    if _ENC is not None:
        return len(_ENC.encode(text))
    return max(1, len(text) // 4)  # ~4 chars/token for English+code


def should_compact(messages: List[dict], *, max_input_tokens: int) -> bool:
    return estimate_tokens(messages) > max_input_tokens
