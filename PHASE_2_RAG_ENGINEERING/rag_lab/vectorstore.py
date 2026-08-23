"""A transparent in-memory vector store (NumPy).

This is the "see the mechanics" store used from notebook 01 onward: embed
chunks, stack into a matrix, retrieve by cosine similarity. Notebook 08 swaps
this for a real vector database (Chroma) with the same ``search`` surface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from rag_lab.chunking import Chunk
from rag_lab.embeddings import embed_texts


@dataclass
class Hit:
    chunk: Chunk
    score: float


class NumpyStore:
    """Cosine-similarity retrieval over a normalized NumPy matrix."""

    def __init__(self, provider: str = "ollama", model: str | None = None):
        self.provider = provider
        self.model = model
        self.chunks: list[Chunk] = []
        self._matrix: np.ndarray | None = None

    def add(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        vecs = embed_texts(
            [c.text for c in chunks], provider=self.provider, model=self.model
        )
        self.chunks.extend(chunks)
        self._matrix = vecs if self._matrix is None else np.vstack([self._matrix, vecs])

    def search(
        self,
        query: str,
        k: int = 4,
        where: Callable[[Chunk], bool] | None = None,
    ) -> list[Hit]:
        """Top-k by cosine similarity. ``where`` optionally filters on metadata."""
        if self._matrix is None:
            return []
        q = embed_texts([query], provider=self.provider, model=self.model)[0]
        scores = self._matrix @ q  # both sides normalized -> cosine
        order = np.argsort(scores)[::-1]
        hits: list[Hit] = []
        for i in order:
            chunk = self.chunks[i]
            if where and not where(chunk):
                continue
            hits.append(Hit(chunk, float(scores[i])))
            if len(hits) >= k:
                break
        return hits

    def context_for(self, query: str, k: int = 4, **kw: Any) -> str:
        """Retrieved chunks formatted for prompt injection, with [source] tags."""
        return "\n\n---\n\n".join(
            f"[{h.chunk.source}] {h.chunk.text}" for h in self.search(query, k=k, **kw)
        )

    def __len__(self) -> int:
        return len(self.chunks)


class ChromaStore:
    """A persistent vector store backed by Chroma (notebook 08).

    Same ``add`` / ``search`` surface as :class:`NumpyStore`, but the index is
    written to disk so a corpus is embedded once and reused across runs. We pass
    our own normalized embeddings so retrieval matches the rest of ``rag_lab``.
    """

    def __init__(
        self,
        collection: str = "rag",
        persist_dir: str | None = None,
        provider: str = "ollama",
        model: str | None = None,
    ):
        import chromadb  # lazy: only needed from notebook 08 onward

        self.provider = provider
        self.model = model
        client = (
            chromadb.PersistentClient(path=persist_dir)
            if persist_dir
            else chromadb.EphemeralClient()
        )
        # Cosine space so scores align with the NumPy store.
        self._col = client.get_or_create_collection(
            collection, metadata={"hnsw:space": "cosine"}
        )

    def add(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        vecs = embed_texts(
            [c.text for c in chunks], provider=self.provider, model=self.model
        )
        start = self._col.count()
        self._col.add(
            ids=[str(start + i) for i in range(len(chunks))],
            embeddings=vecs.tolist(),
            documents=[c.text for c in chunks],
            metadatas=[{"source": c.source, "index": c.index, **_flat(c.meta)} for c in chunks],
        )

    def search(self, query: str, k: int = 4, where: dict[str, Any] | None = None) -> list[Hit]:
        """Top-k by cosine distance. ``where`` is a Chroma metadata filter dict."""
        q = embed_texts([query], provider=self.provider, model=self.model)[0]
        res = self._col.query(
            query_embeddings=[q.tolist()], n_results=k, where=where or None
        )
        hits: list[Hit] = []
        for doc, meta, dist in zip(
            res["documents"][0], res["metadatas"][0], res["distances"][0]
        ):
            chunk = Chunk(
                text=doc,
                source=str(meta.get("source", "?")),
                index=int(meta.get("index", 0)),
                meta=meta,
            )
            hits.append(Hit(chunk, 1.0 - float(dist)))  # cosine distance -> similarity
        return hits

    def __len__(self) -> int:
        return self._col.count()


def _flat(meta: dict[str, Any]) -> dict[str, Any]:
    """Chroma metadata must be scalar; join list values (e.g. tags) into a string."""
    out: dict[str, Any] = {}
    for k, v in meta.items():
        out[k] = ",".join(v) if isinstance(v, list) else v
    return out
