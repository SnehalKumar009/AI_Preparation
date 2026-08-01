"""Provider registry + factory.

    from study_buddy.providers import get_provider
    llm = get_provider("ollama")        # or "openai" | "gemini" | "deepseek"
    resp = llm.chat([{"role": "user", "content": "hi"}])

Swap providers by changing that one string.
"""

from __future__ import annotations

from study_buddy.providers.base import ChatResponse, LLMProvider, Message, Usage

_REGISTRY = {
    "ollama": ("study_buddy.providers.ollama_provider", "OllamaProvider"),
    "openai": ("study_buddy.providers.openai_provider", "OpenAIProvider"),
    "deepseek": ("study_buddy.providers.openai_provider", "DeepSeekProvider"),
    "gemini": ("study_buddy.providers.gemini_provider", "GeminiProvider"),
    "anthropic": ("study_buddy.providers.anthropic_provider", "AnthropicProvider"),
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


__all__ = [
    "get_provider",
    "available_providers",
    "LLMProvider",
    "ChatResponse",
    "Message",
    "Usage",
]
