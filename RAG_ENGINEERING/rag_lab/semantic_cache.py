"""Semantic caching (notebook 27).

Exact-match caches miss "what is RAG?" vs "explain RAG" — same intent, different
string. A semantic cache keys on *meaning*: embed each query, and on a new query
return a cached answer when cosine similarity to a stored query exceeds a threshold.
This cuts cost and latency for the long tail of near-duplicate questions that any
real RAG service receives.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from rag_lab.embeddings import embed_texts


@dataclass
class CacheEntry:
    query: str
    answer: str


@dataclass
class SemanticCache:
    """Embedding-keyed answer cache with a cosine-similarity hit threshold."""

    threshold: float = 0.9
    provider: str = "ollama"
    model: str | None = None
    entries: list[CacheEntry] = field(default_factory=list)
    hits: int = 0
    misses: int = 0
    _matrix: np.ndarray | None = field(default=None, repr=False)

    def _embed(self, text: str) -> np.ndarray:
        return embed_texts([text], provider=self.provider, model=self.model)[0]

    def get(self, query: str) -> str | None:
        """Return a cached answer if a stored query is similar enough, else None."""
        if self._matrix is None:
            self.misses += 1
            return None
        q = self._embed(query)
        scores = self._matrix @ q  # normalized -> cosine
        best = int(np.argmax(scores))
        if scores[best] >= self.threshold:
            self.hits += 1
            return self.entries[best].answer
        self.misses += 1
        return None

    def put(self, query: str, answer: str) -> None:
        """Store a query/answer pair."""
        vec = self._embed(query).reshape(1, -1)
        self.entries.append(CacheEntry(query, answer))
        self._matrix = vec if self._matrix is None else np.vstack([self._matrix, vec])

    def stats(self) -> str:
        total = self.hits + self.misses
        rate = self.hits / total if total else 0.0
        return f"{self.hits} hits / {self.misses} misses ({rate:.0%} hit rate)"
