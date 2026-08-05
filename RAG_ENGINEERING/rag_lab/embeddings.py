"""Embeddings: a thin, normalized wrapper over study_buddy providers.

Phase 1 exposes ``provider.embed(texts, model=...)``. Here we add NumPy
conversion, L2 normalization (so a dot product == cosine similarity), and a
couple of similarity helpers that the notebooks reuse.
"""

from __future__ import annotations

import numpy as np

from study_buddy import get_provider
from study_buddy.providers.base import LLMProvider

# Default embedding model per provider (only providers that embed are listed).
_EMBED_MODELS = {
    "ollama": "nomic-embed-text",
    "openai": "text-embedding-3-small",
}


def _resolve(provider: str | LLMProvider) -> LLMProvider:
    return provider if isinstance(provider, LLMProvider) else get_provider(provider)


def l2_normalize(vecs: np.ndarray) -> np.ndarray:
    """Scale each row to unit length so dot product equals cosine similarity."""
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vecs / norms


def embed_texts(
    texts: list[str],
    provider: str | LLMProvider = "ollama",
    model: str | None = None,
    normalize: bool = True,
) -> np.ndarray:
    """Embed a list of texts into a (n, dim) float32 matrix."""
    llm = _resolve(provider)
    name = getattr(llm, "name", "")
    model = model or _EMBED_MODELS.get(name)
    vecs = np.array(llm.embed(texts, model=model), dtype=np.float32)
    return l2_normalize(vecs) if normalize else vecs


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(a @ b / denom) if denom else 0.0
