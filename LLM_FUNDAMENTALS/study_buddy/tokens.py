"""Token counting helpers (topics 1 & 2).

tiktoken gives exact counts for OpenAI models and a solid approximation for
others, so notebooks can reason about context windows and cost before a call.
"""

from __future__ import annotations

import tiktoken

from study_buddy.providers.base import Message


def _encoding(model: str):
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        # Fallback used by most modern models.
        return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    return len(_encoding(model).encode(text))


def count_message_tokens(messages: list[Message], model: str = "gpt-4o-mini") -> int:
    """Approximate prompt tokens for a chat request (~4 tokens overhead/message)."""
    enc = _encoding(model)
    total = 0
    for m in messages:
        total += 4 + len(enc.encode(m.get("content", "") or ""))
    return total + 2
