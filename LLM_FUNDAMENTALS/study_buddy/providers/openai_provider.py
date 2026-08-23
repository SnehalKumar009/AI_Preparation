"""OpenAI provider (also base class for the OpenAI-compatible DeepSeek API)."""

from __future__ import annotations

from http import client
import json
import time
from typing import Any, Iterator
from urllib import response

from openai import OpenAI, base_url

from study_buddy.config import settings
from study_buddy.providers.base import ChatResponse, LLMProvider, Message, Usage


class OpenAIProvider(LLMProvider):
    name = "openai"
    default_model = settings.openai_chat_model
    default_embed_model = settings.openai_embed_model

    def __init__(self, api_key: str | None = None, base_url: str | None = None) -> None:
        key = api_key or settings.openai_api_key
        if not key:
            raise ValueError("OPENAI_API_KEY is not set in .env")
        self.client = OpenAI(api_key=key, base_url=base_url)

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
        params: dict[str, Any] = {"model": model, "messages": messages}
        if temperature is not None:
            params["temperature"] = temperature
        if top_p is not None:
            params["top_p"] = top_p
        if tools:
            params["tools"] = tools
        if response_format is not None:
            params["response_format"] = response_format
        params.update(kwargs)

        start = time.perf_counter()
        resp = self.client.chat.completions.create(**params)
        latency = time.perf_counter() - start

        choice = resp.choices[0].message
        tool_calls = [
            {
                "name": tc.function.name,
                "arguments": json.loads(tc.function.arguments or "{}"),
            }
            for tc in (choice.tool_calls or [])
        ]
        usage = resp.usage
        return ChatResponse(
            text=choice.content or "",
            model=model,
            provider=self.name,
            usage=Usage(
                input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                output_tokens=getattr(usage, "completion_tokens", 0) or 0,
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
        params: dict[str, Any] = {"model": model, "messages": messages, "stream": True}
        if temperature is not None:
            params["temperature"] = temperature
        if top_p is not None:
            params["top_p"] = top_p
        params.update(kwargs)

        for chunk in self.client.chat.completions.create(**params):
            if not chunk.choices:
                continue
            piece = chunk.choices[0].delta.content
            if piece:
                yield piece

    def embed(self, texts: list[str], model: str | None = None) -> list[list[float]]:
        model = model or self.default_embed_model
        resp = self.client.embeddings.create(model=model, input=texts)
        return [item.embedding for item in resp.data]


class DeepSeekProvider(OpenAIProvider):
    """DeepSeek exposes an OpenAI-compatible endpoint.

    Use model ``deepseek-chat`` (non-reasoning) or ``deepseek-reasoner``.
    """

    name = "deepseek"
    default_model = settings.deepseek_chat_model
    default_embed_model = None

    def __init__(self, api_key: str | None = None) -> None:
        key = api_key or settings.deepseek_api_key
        if not key:
            raise ValueError("DEEPSEEK_API_KEY is not set in .env")
        # Skip OpenAIProvider.__init__ key check; wire the DeepSeek base URL.
        self.client = OpenAI(api_key=key, base_url="https://api.deepseek.com")

    def embed(self, texts: list[str], model: str | None = None) -> list[list[float]]:
        raise NotImplementedError("DeepSeek has no embeddings API; use Ollama or OpenAI")

class OxAlphaProvider(OpenAIProvider):
    name = "oxalpha"
    default_model = settings.oxalpha_chat_model
    default_embed_model = None

    def __init__(self, api_key: str | None = None) -> None:
        key = api_key or settings.oxalpha_api_key
        if not key:
            raise ValueError("OXALPHA_API_KEY is not set in .env")
        # Skip OpenAIProvider.__init__ key check; wire the OxAlpha base URL.
        self.client = OpenAI(base_url="https://openrouter.ai/api/v1",
            api_key=key,
        )

    def embed(self, texts: list[str], model: str | None = None) -> list[list[float]]:
        raise NotImplementedError("OxAlpha has no embeddings API; use Ollama or OpenAI")