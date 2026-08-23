"""Parent-child (small-to-big) retrieval (notebook 13).

Embed small child chunks for precise matching, but hand the generator the larger
parent passage for context. You get the retrieval precision of small chunks with
the surrounding context of big ones.
"""

from __future__ import annotations

from typing import Any

from rag_lab.chunking import Chunk, parent_child_chunks
from rag_lab.vectorstore import Hit, NumpyStore


class ParentChildRetriever:
    """Search over child chunks, return their (deduplicated) parent passages."""

    def __init__(self, provider: str = "ollama", model: str | None = None):
        self._store = NumpyStore(provider=provider, model=model)
        self._parents: dict[tuple[str, int], Chunk] = {}

    def add(self, text: str, source: str = "inline", **kw: Any) -> None:
        parents, children = parent_child_chunks(text, source=source, **kw)
        for parent in parents:
            self._parents[(source, parent.index)] = parent
        self._store.add(children)

    def search(self, query: str, k: int = 4, pool: int = 12) -> list[Hit]:
        """Retrieve child chunks, then return the best parents once each."""
        seen: set[tuple[str, int]] = set()
        out: list[Hit] = []
        for h in self._store.search(query, k=pool):
            pkey = (h.chunk.source, int(h.chunk.meta.get("parent_index", 0)))
            if pkey in seen:
                continue
            seen.add(pkey)
            parent = self._parents.get(pkey)
            if parent is not None:
                out.append(Hit(parent, h.score))
            if len(out) >= k:
                break
        return out

    def __len__(self) -> int:
        return len(self._store)
