"""Sentence-window / small-to-big retrieval (notebook 23).

A tension in chunking: small chunks match precisely but read out of context; large
chunks give context but dilute the match. Sentence-window retrieval gets both —
index one sentence at a time (precise matching), but at answer time return that
sentence plus its neighbours (the "window") so the generator sees full context.
Also called small-to-big: retrieve small, return big.
"""

from __future__ import annotations

from typing import Any, Callable

from rag_lab.chunking import Chunk, _sentences
from rag_lab.vectorstore import Hit, NumpyStore


class SentenceWindowRetriever:
    """Embed individual sentences; return each match widened to a sentence window."""

    def __init__(self, window: int = 1, provider: str = "ollama", model: str | None = None):
        self.window = window
        self._store = NumpyStore(provider=provider, model=model)
        self._sentences: list[str] = []

    def add(self, chunks: list[Chunk]) -> None:
        """Explode each chunk into per-sentence chunks tagged with their position."""
        sent_chunks: list[Chunk] = []
        for c in chunks:
            print("================================")
            print("Adding chunk:", c.text)
            for s in _sentences(c.text):
                pos = len(self._sentences)
                self._sentences.append(s)
                sent_chunks.append(Chunk(s, source=c.source, index=pos, meta={"pos": pos}))
                print("    Sentence so far:")
                for i, sent in enumerate(self._sentences):
                    print(f"        {i}: {sent}")
            print("================================")
        self._store.add(sent_chunks)

    def _window_text(self, pos: int) -> str:
        lo = max(0, pos - self.window)
        hi = min(len(self._sentences), pos + self.window + 1)
        return " ".join(self._sentences[lo:hi])

    def search(self, query: str, k: int = 4, where: Callable[[Chunk], bool] | None = None) -> list[Hit]:
        """Retrieve on single sentences, then widen each hit to its window."""
        hits = self._store.search(query, k=k, where=where)
        print("=========search start===========")
        for h in hits:
            print(f"Hit: {h.chunk.text} (score: {h.score})")
        print("=========search end===========")
        print("=========widening start===========")
        widened: list[Hit] = []
        for h in hits:
            pos = int(h.chunk.meta.get("pos", h.chunk.index))
            print(f"index : {h.chunk.index}")
            widened.append(
                Hit(
                    Chunk(self._window_text(pos), source=h.chunk.source, index=pos,
                          meta={"matched": h.chunk.text}),
                    h.score,
                )
            )
            print(f"Widened hit: {widened[-1].chunk.text}")
        print("=========widening end===========")
        return widened

    def __len__(self) -> int:
        return len(self._sentences)
