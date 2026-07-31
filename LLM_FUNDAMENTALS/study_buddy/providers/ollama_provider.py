"""Local Ollama provider."""

from __future__ import annotations

import time
from typing import Any, Iterator

import ollama

from study_buddy.config import settings
from study_buddy.providers.base import ChatResponse, LLMProvider, Message, Usage


class OllamaProvider(LLMProvider):
    name = "ollama"
    default_model = settings.ollama_chat_model
    default_embed_model = settings.ollama_embed_model

    def __init__(self, host: str | None = None) -> None:
        self.client = ollama.Client(host=host or settings.ollama_host)

    def _options(self, temperature: float | None, top_p: float | None) -> dict[str, Any]:
        opts: dict[str, Any] = {}
        if temperature is not None:
            opts["temperature"] = temperature
        if top_p is not None:
            opts["top_p"] = top_p
        return opts

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
        model = model or self.default_model
        # Ollama accepts a JSON schema (or "json") via the `format` arg.
        fmt = None
        if response_format is not None:
            fmt = response_format.get("schema", "json")

        start = time.perf_counter()
        resp = self.client.chat(
            model=model,
            messages=messages,
            options=self._options(temperature, top_p),
            tools=tools,
            format=fmt,
            **kwargs,
        )
        latency = time.perf_counter() - start

        msg = resp.get("message", {})
        tool_calls = [
            {
                "name": tc["function"]["name"],
                "arguments": tc["function"]["arguments"],
            }
            for tc in (msg.get("tool_calls") or [])
        ]
        return ChatResponse(
            text=msg.get("content", ""),
            model=model,
            provider=self.name,
            usage=Usage(
                input_tokens=resp.get("prompt_eval_count", 0) or 0,
                output_tokens=resp.get("eval_count", 0) or 0,
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
        **kwargs: Any,
    ) -> Iterator[str]:
        model = model or self.default_model
        for chunk in self.client.chat(
            model=model,
            messages=messages,
            options=self._options(temperature, top_p),
            stream=True,
            **kwargs,
        ):
            piece = chunk.get("message", {}).get("content", "")
            if piece:
                yield piece

    def embed(self, texts: list[str], model: str | None = None) -> list[list[float]]:
        model = model or self.default_embed_model
        resp = self.client.embed(model=model, input=texts)
        return resp["embeddings"]
