"""Hybrid retrieval with Reciprocal Rank Fusion (notebook 10).

Dense search finds paraphrases; BM25 finds exact terms. Hybrid runs both and
fuses their ranked lists with RRF: each hit scores ``1 / (rrf_k + rank)`` in each
list and the per-hit scores are summed. RRF needs no score calibration between the
two systems, which makes it a robust default.
"""

from __future__ import annotations

from typing import Any

from rag_lab.chunking import Chunk
from rag_lab.vectorstore import Hit


def _key(chunk: Chunk) -> tuple[str, int]:
    return (chunk.source, chunk.index)


class HybridRetriever:
    """Fuse a dense store and a sparse (BM25) store via Reciprocal Rank Fusion.

    Both stores only need a ``search(query, k) -> list[Hit]`` method, so any pair
    of the ``rag_lab`` retrievers can be combined.
    """

    def __init__(self, dense: Any, sparse: Any, rrf_k: int = 60):
        self.dense = dense
        self.sparse = sparse
        self.rrf_k = rrf_k

    def search(self, query: str, k: int = 4, pool: int = 10) -> list[Hit]:
        """Top-k after fusing each retriever's top-``pool`` results with RRF."""
        fused: dict[tuple[str, int], float] = {}
        chunks: dict[tuple[str, int], Chunk] = {}
        for hits in (self.dense.search(query, k=pool), self.sparse.search(query, k=pool)):
            for rank, h in enumerate(hits):
                key = _key(h.chunk)
                fused[key] = fused.get(key, 0.0) + 1.0 / (self.rrf_k + rank)
                chunks[key] = h.chunk
        ranked = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
        return [Hit(chunks[key], score) for key, score in ranked[:k]]
