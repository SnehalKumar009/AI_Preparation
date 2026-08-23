"""Google Gemini provider (google-genai SDK)."""

from __future__ import annotations

import time
from typing import Any, Iterator

from google import genai
from google.genai import types

from study_buddy.config import settings
from study_buddy.providers.base import ChatResponse, LLMProvider, Message, Usage


def _to_gemini(messages: list[Message]) -> tuple[str | None, list[types.Content]]:
    """Split OpenAI-style messages into (system_instruction, gemini contents)."""
    system_parts: list[str] = []
    contents: list[types.Content] = []
    for m in messages:
        role = m["role"]
        text = m.get("content", "") or ""
        if role == "system":
            system_parts.append(text)
            continue
        gem_role = "model" if role == "assistant" else "user"
        contents.append(types.Content(role=gem_role, parts=[types.Part(text=text)]))
    system = "\n".join(system_parts) if system_parts else None
    return system, contents


class GeminiProvider(LLMProvider):
    name = "gemini"
    default_model = settings.gemini_chat_model
    default_embed_model = "text-embedding-004"

    def __init__(self, api_key: str | None = None) -> None:
        key = api_key or settings.gemini_api_key
        if not key:
            raise ValueError("GEMINI_API_KEY is not set in .env")
        self.client = genai.Client(api_key=key)

    def _config(
        self,
        system: str | None,
        temperature: float | None,
        top_p: float | None,
        response_format: dict[str, Any] | None,
    ) -> types.GenerateContentConfig:
        cfg: dict[str, Any] = {}
        if system:
            cfg["system_instruction"] = system
        if temperature is not None:
            cfg["temperature"] = temperature
        if top_p is not None:
            cfg["top_p"] = top_p
        if response_format is not None:
            cfg["response_mime_type"] = "application/json"
            if "schema" in response_format:
                cfg["response_schema"] = response_format["schema"]
        return types.GenerateContentConfig(**cfg)

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
        system, contents = _to_gemini(messages)

        start = time.perf_counter()
        resp = self.client.models.generate_content(
            model=model,
            contents=contents,
            config=self._config(system, temperature, top_p, response_format),
        )
        latency = time.perf_counter() - start

        um = resp.usage_metadata
        return ChatResponse(
            text=resp.text or "",
            model=model,
            provider=self.name,
            usage=Usage(
                input_tokens=getattr(um, "prompt_token_count", 0) or 0,
                output_tokens=getattr(um, "candidates_token_count", 0) or 0,
            ),
            latency_s=latency,
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
        system, contents = _to_gemini(messages)
        for chunk in self.client.models.generate_content_stream(
            model=model,
            contents=contents,
            config=self._config(system, temperature, top_p, None),
        ):
            if chunk.text:
                yield chunk.text

    def embed(self, texts: list[str], model: str | None = None) -> list[list[float]]:
        model = model or self.default_embed_model
        resp = self.client.models.embed_content(model=model, contents=texts)
        return [e.values for e in resp.embeddings]
