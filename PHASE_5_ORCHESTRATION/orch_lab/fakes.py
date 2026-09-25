"""Deterministic stand-ins for teaching (a scripted model, a simulated crash).

Real models are non-deterministic, so a demo that depends on "the model skips a
step" or "the process dies on step 3" might not happen on the day you run it.
These fakes make those situations happen **every time**, so a notebook can show
a failure mode on purpose; each notebook then repeats the idea with a real model.

- :class:`ScriptedLLM` — a provider that replays a fixed list of turns. It
  implements the same interface as ``study_buddy`` providers, so it drops into
  the Phase 3 :class:`~agents_lab.loop.Agent` unchanged.
- :class:`ProcessCrash` — an exception that stands in for ``kill -9``, an OOM,
  or a redeploy. It derives from ``BaseException`` so the tool registry's
  ``except Exception`` safety net can't catch it, just as it can't catch a
  real process death.
"""

from __future__ import annotations

from typing import Any, Iterator

from study_buddy.providers.base import ChatResponse, LLMProvider, Message, Usage

# One scripted turn: a string is a final answer; a list of
# {"name": ..., "arguments": {...}} dicts is a batch of tool calls.
Turn = str | list[dict[str, Any]]


class ProcessCrash(BaseException):
    """Simulates the process dying mid-run (not a normal, catchable error)."""


class ScriptedLLM(LLMProvider):
    """A fake model that returns pre-written turns in order.

    Each ``chat`` call consumes the next turn and reports a fixed token usage, so
    notebooks can count how many model calls (and tokens) a run cost — and how
    many were wasted when it had to start over.
    """

    name = "scripted"
    default_model = "scripted"

    def __init__(
        self,
        turns: list[Turn],
        input_tokens: int = 600,
        output_tokens: int = 60,
    ) -> None:
        self.turns = list(turns)
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.calls = 0

    def chat(
        self,
        messages: list[Message],
        model: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        tools: list[dict[str, Any]] | None = None,
        response_format: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> ChatResponse:
        if self.calls >= len(self.turns):
            turn: Turn = "(script exhausted) Done."
        else:
            turn = self.turns[self.calls]
        self.calls += 1
        usage = Usage(self.input_tokens, self.output_tokens)
        if isinstance(turn, str):
            return ChatResponse(turn, self.default_model, self.name, usage)
        return ChatResponse("", self.default_model, self.name, usage, tool_calls=list(turn))

    def stream(
        self,
        messages: list[Message],
        model: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        yield self.chat(messages).text

    @property
    def tokens_used(self) -> int:
        return self.calls * (self.input_tokens + self.output_tokens)
