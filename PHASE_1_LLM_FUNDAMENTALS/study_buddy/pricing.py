"""Pricing table loader + cost estimation from token usage."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

from study_buddy.config import settings
from study_buddy.providers.base import Usage


@lru_cache(maxsize=1)
def load_pricing(path: str | Path | None = None) -> dict[str, dict[str, float]]:
    """Load per-model USD/1M-token rates from pricing.yaml (cached)."""
    path = Path(path) if path else settings.pricing_file
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def estimate_cost(model: str, usage: Usage, path: str | Path | None = None) -> float:
    """Return USD cost for a call. Unknown models cost 0 (with no crash)."""
    rates = load_pricing(path).get(model)
    if not rates:
        return 0.0
    return (
        usage.input_tokens / 1_000_000 * rates.get("input", 0.0)
        + usage.output_tokens / 1_000_000 * rates.get("output", 0.0)
    )
