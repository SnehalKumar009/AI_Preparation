"""Helpers shared by this section's notebooks, and by nothing outside it.

Notebook 03 explains how the Agents SDK talks to models and builds this by hand.
After that, every notebook simply does ``from common import get_model``.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

SECTION_DIR = Path(__file__).resolve().parent
load_dotenv(SECTION_DIR / ".env")          # this section's own .env (optional)

DEFAULT_MODELS = {"ollama": "qwen3-coder:30b", "openai": "gpt-4.1-mini"}


def _configure_tracing() -> None:
    """The SDK uploads traces to OpenAI by default. Without a key that fails, so turn it off."""
    from agents import set_tracing_disabled

    set_tracing_disabled(not os.getenv("OPENAI_API_KEY"))


_configure_tracing()


def get_model(provider: str | None = None):
    """Return an Agents-SDK model for ``provider`` (default: $MODEL_PROVIDER or ollama).

    * ollama → ``OpenAIChatCompletionsModel`` pointed at Ollama's OpenAI-compatible API
    * openai → the model name as a string; the SDK uses OpenAI's Responses API
    """
    provider = (provider or os.getenv("MODEL_PROVIDER") or "ollama").lower()
    name = os.getenv(f"{provider.upper()}_MODEL") or DEFAULT_MODELS[provider]

    if provider == "ollama":
        from agents import OpenAIChatCompletionsModel
        from openai import AsyncOpenAI

        base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
        client = AsyncOpenAI(base_url=f"{base}/v1", api_key="ollama")   # Ollama ignores the key
        return OpenAIChatCompletionsModel(model=name, openai_client=client)

    if provider == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set (add it to 02_OPENAI_AGENTS_SDK/.env).")
        return name

    raise ValueError(f"Unknown provider {provider!r}; use 'ollama' or 'openai'.")


def has_key(provider: str) -> bool:
    return bool(os.getenv(f"{provider.upper()}_API_KEY"))


# ---------------------------------------------------------------------------
# Local models sometimes write a tool call as plain TEXT (e.g. "<function=search…")
# instead of a real, structured tool call. The Runner then sees no tool call and
# ends the run with that text as the "answer". Notebook 11 measures and explains it.
LEAK_MARKERS = ("<function=", "<tool_call>", "</tool_call>")


def leaked_tool_call(output) -> bool:
    """True if a final output looks like a tool call written as text."""
    return isinstance(output, str) and any(marker in output for marker in LEAK_MARKERS)


async def run_agent(agent, input, *, retries: int = 2, **kwargs):
    """``Runner.run`` plus a retry when the model leaks a tool call as text."""
    from agents import Runner

    for attempt in range(retries + 1):
        result = await Runner.run(agent, input, **kwargs)
        if not leaked_tool_call(result.final_output) or attempt == retries:
            return result
    return result
