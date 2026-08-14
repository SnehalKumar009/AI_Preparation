"""Keyword search with BM25 (notebook 09).

Dense vector search matches *meaning*; BM25 matches *terms*. It scores a passage
by how often the query words appear, dampened by document length and by how common
each term is across the corpus — so it shines on exact tokens (error codes,
function names, rare proper nouns) that embeddings blur together. The store mirrors
``NumpyStore``'s ``add`` / ``search`` surface so it drops straight into hybrid
retrieval (notebook 10).
"""

from __future__ import annotations

import re
from typing import Callable

from rag_lab.chunking import Chunk
from rag_lab.vectorstore import Hit

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase word/number tokens — the minimal tokenizer BM25 needs."""
    return _TOKEN.findall(text.lower())


class BM25Store:
    """Sparse, term-frequency retrieval over the corpus (rank_bm25 backend)."""

    def __init__(self) -> None:
        self.chunks: list[Chunk] = []
        self._tokens: list[list[str]] = []
        self._bm25 = None  # built lazily; invalidated whenever chunks change

    def add(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        self.chunks.extend(chunks)
        self._tokens.extend(tokenize(c.text) for c in chunks)
        self._bm25 = None

    def _ensure_index(self) -> None:
        if self._bm25 is None and self._tokens:
            from rank_bm25 import BM25Okapi  # lazy: only needed from notebook 09

            self._bm25 = BM25Okapi(self._tokens)

    def search(
        self,
        query: str,
        k: int = 4,
        where: Callable[[Chunk], bool] | None = None,
    ) -> list[Hit]:
        """Top-k by BM25 score. ``where`` optionally filters on metadata."""
        self._ensure_index()
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        hits: list[Hit] = []
        for i in order:
            chunk = self.chunks[i]
            if where and not where(chunk):
                continue
            hits.append(Hit(chunk, float(scores[i])))
            if len(hits) >= k:
                break
        return hits

    def __len__(self) -> int:
        return len(self.chunks)
