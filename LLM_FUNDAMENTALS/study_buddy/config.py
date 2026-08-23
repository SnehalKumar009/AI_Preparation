"""Central configuration: loads .env and exposes default models per role."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the project root (parent of the study_buddy package).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    # Ollama
    ollama_host: str = field(default_factory=lambda: os.getenv("OLLAMA_HOST", "http://localhost:11434"))

    # Cloud keys (may be None until the user fills .env)
    openai_api_key: str | None = field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    gemini_api_key: str | None = field(default_factory=lambda: os.getenv("GEMINI_API_KEY"))
    deepseek_api_key: str | None = field(default_factory=lambda: os.getenv("DEEPSEEK_API_KEY"))
    anthropic_api_key: str | None = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY"))
    oxalpha_api_key: str | None = field(default_factory=lambda: os.getenv("OXALPHA_API_KEY"))

    # Default models per role (see plan)
    ollama_chat_model: str = "qwen3-coder:30b"
    ollama_embed_model: str = "nomic-embed-text"
    openai_chat_model: str = "gpt-4o-mini"
    openai_embed_model: str = "text-embedding-3-small"
    gemini_chat_model: str = "gemini-2.0-flash"
    deepseek_chat_model: str = "deepseek-chat"
    deepseek_reasoner_model: str = "deepseek-reasoner"
    anthropic_chat_model: str = "claude-haiku-4-5-20251001"
    oxalpha_chat_model: str = "stealth/ox-alpha"

    # Filesystem
    notes_dir: Path = PROJECT_ROOT / "notes"
    pricing_file: Path = PROJECT_ROOT / "study_buddy" / "pricing.yaml"
    rag_cache_dir: Path = PROJECT_ROOT / "study_buddy" / ".rag_cache"


settings = Settings()
