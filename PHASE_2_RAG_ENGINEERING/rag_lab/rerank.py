"""Reranking: precision after recall (notebook 11).

A first-stage retriever favors recall — return many candidates cheaply. A reranker
then favors precision, scoring each candidate against the query with a more
expensive model. The default here is an **LLM-as-judge**: the model rates each
passage 0-10 for relevance and we reorder. If ``sentence-transformers`` is
installed, :func:`cross_encoder_rerank` offers a classic cross-encoder instead.
"""

from __future__ import annotations

import re
from typing import Any

from study_buddy import get_provider

from rag_lab.vectorstore import Hit

_SCORE = re.compile(r"\d+(?:\.\d+)?")

_PROMPT = (
    "Rate how well the PASSAGE answers the QUERY on a scale of 0 to 10, where 10 is "
    "a perfect answer and 0 is irrelevant. Reply with only the number.\n\n"
    "QUERY: {query}\n\nPASSAGE: {passage}\n\nScore:"
)


class Reranker:
    """Reorder candidate hits by LLM-judged relevance to the query."""

    def __init__(
        self,
        provider: str = "ollama",
        model: str | None = None,
        tracker: Any | None = None,
    ):
        self.provider = provider
        self.model = model
        self.tracker = tracker  # optional study_buddy CostTracker
        self._llm = None

    def _score(self, query: str, text: str) -> float:
        if self._llm is None:
            self._llm = get_provider(self.provider)
        resp = self._llm.chat(
            [{"role": "user", "content": _PROMPT.format(query=query, passage=text)}],
            model=self.model,
            temperature=0.0,
        )
        if self.tracker is not None:
            self.tracker.add(resp)
        m = _SCORE.search(resp.text)
        return float(m.group()) if m else 0.0

    def rerank(self, query: str, hits: list[Hit], k: int = 4) -> list[Hit]:
        """Return the top-``k`` hits after LLM relevance scoring."""
        scored = [Hit(h.chunk, self._score(query, h.chunk.text)) for h in hits]
        scored.sort(key=lambda h: h.score, reverse=True)
        return scored[:k]


def cross_encoder_available() -> bool:
    """True if sentence-transformers (and a cross-encoder) can be imported."""
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        return False
    return True


def cross_encoder_rerank(
    query: str,
    hits: list[Hit],
    k: int = 4,
    model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
) -> list[Hit]:
    """Rerank with a cross-encoder (requires ``sentence-transformers``)."""
    from sentence_transformers import CrossEncoder

    ce = CrossEncoder(model)
    scores = ce.predict([(query, h.chunk.text) for h in hits])
    ranked = sorted(zip(hits, scores), key=lambda hs: hs[1], reverse=True)
    return [Hit(h.chunk, float(s)) for h, s in ranked[:k]]
