"""Short-term / working memory for an agent (notebook 13).

Agents need somewhere to keep what they've learned *within* a task: a running
scratchpad of notes and a rolling window of conversation that stays under a token
budget. This is deliberately minimal — the full memory taxonomy (long-term,
semantic, episodic, procedural) is Phase 6. Here we cover just the short-term
working memory the loop uses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Scratchpad:
    """A running list of notes the agent accumulates during a task."""

    notes: list[str] = field(default_factory=list)

    def add(self, note: str) -> None:
        self.notes.append(note.strip())

    def render(self) -> str:
        return "\n".join(f"- {n}" for n in self.notes) or "(empty)"

    def clear(self) -> None:
        self.notes.clear()

    def __len__(self) -> int:
        return len(self.notes)


class ShortTermMemory:
    """A rolling window of chat messages capped by an approximate token budget."""

    def __init__(self, max_tokens: int = 2000, model: str = "gpt-4o-mini") -> None:
        self.max_tokens = max_tokens
        self.model = model
        self.messages: list[dict[str, Any]] = []

    def add(self, role: str, content: str) -> None:
        self.messages.append({"role": role, "content": content})
        self._trim()

    def _count(self, msgs: list[dict[str, Any]]) -> int:
        from study_buddy import count_message_tokens

        return count_message_tokens(msgs, model=self.model)

    def _trim(self) -> None:
        # Drop oldest non-system messages until under budget.
        while len(self.messages) > 1 and self._count(self.messages) > self.max_tokens:
            drop = next((i for i, m in enumerate(self.messages) if m["role"] != "system"), None)
            if drop is None:
                break
            self.messages.pop(drop)

    def window(self) -> list[dict[str, Any]]:
        return list(self.messages)
