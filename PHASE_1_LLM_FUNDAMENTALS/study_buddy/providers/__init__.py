"""Provider registry + factory.

    from study_buddy.providers import get_provider
    llm = get_provider("ollama")        # or "openai" | "gemini" | "deepseek"
    resp = llm.chat([{"role": "user", "content": "hi"}])

Swap providers by changing that one string.
"""

from __future__ import annotations
from typing import Any

from study_buddy.providers.base import ChatResponse, LLMProvider, Message, Usage

_REGISTRY = {
    "ollama": ("study_buddy.providers.ollama_provider", "OllamaProvider"),
    "openai": ("study_buddy.providers.openai_provider", "OpenAIProvider"),
    "deepseek": ("study_buddy.providers.openai_provider", "DeepSeekProvider"),
    "gemini": ("study_buddy.providers.gemini_provider", "GeminiProvider"),
    "anthropic": ("study_buddy.providers.anthropic_provider", "AnthropicProvider"),
    "oxalpha": ("study_buddy.providers.openai_provider", "OxAlphaProvider"),
}


def available_providers() -> list[str]:
    return list(_REGISTRY)


def get_provider(name: str, **kwargs) -> LLMProvider:
    """Instantiate a provider by name. Imports lazily so a missing cloud SDK
    or key only fails if you actually use that provider."""
    key = name.lower()
    if key not in _REGISTRY:
        raise ValueError(f"Unknown provider '{name}'. Options: {available_providers()}")
    module_path, class_name = _REGISTRY[key]
    module = __import__(module_path, fromlist=[class_name])
    return getattr(module, class_name)(**kwargs)


def test_provider() -> None:
    """Smoke test: confirm we can reach OpenRouter's OpenAI-compatible API.

    Uses the standard OpenAI SDK (already a dependency) against OpenRouter's
    base URL. Prefers a key from the environment (OPENROUTER_API_KEY /
    OXALPHA_API_KEY); the hardcoded fallback is only here so the test still
    runs without a .env.
    """
    import os

    from openai import OpenAI

    api_key = (
        os.getenv("OPENROUTER_API_KEY")
        or os.getenv("OXALPHA_API_KEY")
        or "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"
    )
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )
    response = client.chat.completions.create(
        model="stealth/ox-alpha",
        messages=[{"role": "user", "content": "Hi How are you?"}],
    )

    print("MODEL:   ", response.model)
    print("USAGE:   ", response.usage)
    print("CONTENT: ", response.choices[0].message.content)
    

__all__ = [
    "get_provider",
    "available_providers",
    "LLMProvider",
    "ChatResponse",
    "Message",
    "Usage",
    "test_provider",
]
