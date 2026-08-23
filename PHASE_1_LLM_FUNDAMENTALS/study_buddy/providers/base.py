"""Provider interface + shared data types.

Every provider speaks the same OpenAI-style message format
(list of {"role", "content"}) and returns token usage so cost is uniform.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterator

# A chat message in the shared, provider-neutral format.
Message = dict[str, Any]


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class ChatResponse:
    text: str
    model: str
    provider: str
    usage: Usage = field(default_factory=Usage)
    latency_s: float = 0.0
    # Tool calls the model requested, if any (normalized shape).
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    raw: Any = None


class LLMProvider(ABC):
    """Common surface for local and cloud LLMs."""

    #: Short provider id, e.g. "ollama", "openai".
    name: str = "base"
    #: Default chat model for this provider.
    default_model: str = ""
    #: Default embedding model for this provider.
    default_embed_model: str | None = None

    @abstractmethod
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
        """Single-shot completion. Returns text + token usage."""

    @abstractmethod
    def stream(
        self,
        messages: list[Message],
        model: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        """Yield text chunks as they arrive."""

    def embed(self, texts: list[str], model: str | None = None) -> list[list[float]]:
        """Return one embedding vector per input text."""
        raise NotImplementedError(f"{self.name} does not support embeddings")
