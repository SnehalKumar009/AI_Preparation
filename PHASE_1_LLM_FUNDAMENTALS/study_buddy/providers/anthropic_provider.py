"""Anthropic (Claude) provider."""

from __future__ import annotations

import time
from typing import Any, Iterator

from anthropic import Anthropic

from study_buddy.config import settings
from study_buddy.providers.base import ChatResponse, LLMProvider, Message, Usage

# Anthropic requires an explicit max output token budget.
_DEFAULT_MAX_TOKENS = 1024


def _split_system(messages: list[Message]) -> tuple[str | None, list[Message]]:
    """Anthropic takes the system prompt as a top-level arg, not a message."""
    system_parts = [m["content"] for m in messages if m["role"] == "system"]
    convo = [
        {"role": m["role"], "content": m.get("content", "") or ""}
        for m in messages
        if m["role"] in ("user", "assistant")
    ]
    system = "\n".join(system_parts) if system_parts else None
    return system, convo


def _to_anthropic_tools(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
    """Convert OpenAI-style tool schemas to Anthropic's shape."""
    if not tools:
        return None
    converted = []
    for t in tools:
        fn = t.get("function", t)
        converted.append(
            {
                "name": fn["name"],
                "description": fn.get("description", ""),
                "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
            }
        )
    return converted


class AnthropicProvider(LLMProvider):
    name = "anthropic"
    default_model = settings.anthropic_chat_model
    default_embed_model = None  # Anthropic has no embeddings API

    def __init__(self, api_key: str | None = None) -> None:
        key = api_key or settings.anthropic_api_key
        if not key:
            raise ValueError("ANTHROPIC_API_KEY is not set in .env")
        self.client = Anthropic(api_key=key)

    def chat(
        self,
        messages: list[Message],
        model: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        tools: list[dict[str, Any]] | None = None,
        response_format: dict[str, Any] | None = None,
        max_tokens: int = _DEFAULT_MAX_TOKENS,
        **kwargs: Any,
    ) -> ChatResponse:
        model = model or self.default_model
        system, convo = _split_system(messages)
        params: dict[str, Any] = {"model": model, "messages": convo, "max_tokens": max_tokens}
        if system:
            params["system"] = system
        if temperature is not None:
            params["temperature"] = temperature
        if top_p is not None:
            params["top_p"] = top_p
        anthropic_tools = _to_anthropic_tools(tools)
        if anthropic_tools:
            params["tools"] = anthropic_tools
        params.update(kwargs)

        start = time.perf_counter()
        resp = self.client.messages.create(**params)
        latency = time.perf_counter() - start

        text = "".join(b.text for b in resp.content if b.type == "text")
        tool_calls = [
            {"name": b.name, "arguments": b.input}
            for b in resp.content
            if b.type == "tool_use"
        ]
        return ChatResponse(
            text=text,
            model=model,
            provider=self.name,
            usage=Usage(
                input_tokens=resp.usage.input_tokens,
                output_tokens=resp.usage.output_tokens,
            ),
            latency_s=latency,
            tool_calls=tool_calls,
            raw=resp,
        )

    def stream(
        self,
        messages: list[Message],
        model: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int = _DEFAULT_MAX_TOKENS,
        **kwargs: Any,
    ) -> Iterator[str]:
        model = model or self.default_model
        system, convo = _split_system(messages)
        params: dict[str, Any] = {"model": model, "messages": convo, "max_tokens": max_tokens}
        if system:
            params["system"] = system
        if temperature is not None:
            params["temperature"] = temperature
        if top_p is not None:
            params["top_p"] = top_p
        params.update(kwargs)

        with self.client.messages.stream(**params) as stream:
            for text in stream.text_stream:
                if text:
                    yield text
