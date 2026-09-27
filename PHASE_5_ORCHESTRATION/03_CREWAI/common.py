"""Helpers shared by this section's notebooks, and by nothing outside it.

Import this module BEFORE importing crewai: it switches off CrewAI's anonymous
usage telemetry, which is read when crewai is first imported.
Notebook 03 explains how CrewAI talks to models and builds ``get_llm`` by hand.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

SECTION_DIR = Path(__file__).resolve().parent
load_dotenv(SECTION_DIR / ".env")                   # this section's own .env (optional)

# CrewAI sends anonymous telemetry by default. Keep everything local.
os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")
os.environ.setdefault("CREWAI_STORAGE_DIR", str(SECTION_DIR / "notebooks" / "checkpoints" / "crewai_storage"))

DEFAULT_MODELS = {"ollama": "qwen3-coder:30b", "openai": "gpt-4.1-mini"}


def get_llm(provider: str | None = None, temperature: float = 0.0, **kwargs):
    """Return a CrewAI ``LLM`` for ``provider`` (default: $MODEL_PROVIDER or ollama).

    CrewAI names models as "<provider>/<model>". For Ollama it uses its native
    OpenAI-compatible client; no LiteLLM needed.
    """
    from crewai import LLM

    provider = (provider or os.getenv("MODEL_PROVIDER") or "ollama").lower()
    name = os.getenv(f"{provider.upper()}_MODEL") or DEFAULT_MODELS[provider]

    if provider == "ollama":
        base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
        return LLM(model=f"ollama/{name}", base_url=base, temperature=temperature, **kwargs)
    if provider == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set (add it to 03_CREWAI/.env).")
        return LLM(model=f"openai/{name}", temperature=temperature, **kwargs)
    raise ValueError(f"Unknown provider {provider!r}; use 'ollama' or 'openai'.")


def embedder_config() -> dict:
    """Embedder settings for CrewAI memory, using a local Ollama embedding model."""
    base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    return {"provider": "ollama",
            "config": {"model_name": os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text"),
                       "url": f"{base}/api/embeddings"}}


def has_key(provider: str) -> bool:
    return bool(os.getenv(f"{provider.upper()}_API_KEY"))


def quiet_console(quiet: bool = True) -> None:
    """Hide (or show again) CrewAI's rich console panels (memory saves, flow pauses, …).

    CrewAI prints these through a global event listener whose formatter is always
    verbose. The notebooks hide them so the teaching output stays readable;
    notebook 10 turns them back on to show what they contain.
    """
    from crewai.events.event_listener import EventListener

    EventListener().formatter.verbose = not quiet


# ---------------------------------------------------------------------------
# Local models sometimes write a tool call as plain TEXT ("<function=search…")
# instead of a real tool call. CrewAI then treats that text as the task's answer,
# and later tasks build on it. Notebook 11 shows it; use this as a task guardrail.
LEAK_MARKERS = ("<function=", "<tool_call>", "</tool_call>")


def reject_leaked_tool_calls(output):
    """Task guardrail: send the agent back if its answer is a tool call written as text."""
    if any(marker in output.raw for marker in LEAK_MARKERS):
        return False, ("Your answer contained a tool call written as text. Call the tool properly, "
                       "then give the final answer in plain prose.")
    return True, output
