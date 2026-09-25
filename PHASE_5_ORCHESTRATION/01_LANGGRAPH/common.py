"""Helpers shared by this section's notebooks, and by nothing outside it.

Notebook 03 explains how LangChain chat models work and builds this function by
hand. After that, every notebook simply does ``from common import get_model``.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

SECTION_DIR = Path(__file__).resolve().parent
load_dotenv(SECTION_DIR / ".env")          # this section's own .env (optional)

DEFAULT_MODELS = {
    "ollama": "qwen3-coder:30b",
    "openai": "gpt-4.1-mini",
    "anthropic": "claude-haiku-4-5",
}


def get_model(provider: str | None = None, temperature: float = 0.0, **kwargs):
    """Return a LangChain chat model for ``provider`` (default: $MODEL_PROVIDER or ollama).

    Every provider returns the same interface (``invoke``, ``stream``,
    ``bind_tools``, ``with_structured_output``), so graphs never care which one
    they're given.
    """
    provider = (provider or os.getenv("MODEL_PROVIDER") or "ollama").lower()
    model = os.getenv(f"{provider.upper()}_MODEL") or DEFAULT_MODELS[provider]

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=model,
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            temperature=temperature,
            **kwargs,
        )

    from langchain.chat_models import init_chat_model

    if not os.getenv(f"{provider.upper()}_API_KEY"):
        raise RuntimeError(f"{provider.upper()}_API_KEY is not set (add it to 01_LANGGRAPH/.env).")
    return init_chat_model(f"{provider}:{model}", temperature=temperature, **kwargs)


def has_key(provider: str) -> bool:
    """True if a cloud provider's API key is configured."""
    return bool(os.getenv(f"{provider.upper()}_API_KEY"))
