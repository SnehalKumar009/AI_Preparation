"""Query transformation (notebook 20).

The user's raw question is often a poor search query. Rewriting it before
retrieval lifts recall. Three classic transforms live here:

- **HyDE** — hallucinate a hypothetical answer and retrieve with *that* (a doc
  that looks like the answer sits near the real answer in embedding space).
- **Step-back** — ask a broader background question first, so foundational
  context is retrieved alongside the specific one.
- **Multi-query / RAG-Fusion** — generate several paraphrases, retrieve each, and
  fuse the ranked lists with Reciprocal Rank Fusion.
"""

from __future__ import annotations

from typing import Any

from study_buddy import get_provider

from rag_lab.vectorstore import Hit

_HYDE = (
    "Write a short, factual paragraph that could plausibly answer the question "
    "below. It is fine to invent specifics — we only use it to search.\n\n"
    "Question: {query}\n\nParagraph:"
)
_STEP_BACK = (
    "Given the specific question below, write ONE broader 'step-back' question "
    "whose answer gives useful background. Reply with only the question.\n\n"
    "Question: {query}\n\nStep-back question:"
)
_MULTI = (
    "Generate {n} alternative phrasings of the search query below, each on its own "
    "line, no numbering. Vary vocabulary and specificity.\n\nQuery: {query}\n\nVariants:"
)


def _chat(prompt: str, provider: str, model: str | None, tracker: Any | None) -> str:
    resp = get_provider(provider).chat(
        [{"role": "user", "content": prompt}], model=model, temperature=0.3
    )
    if tracker is not None:
        tracker.add(resp)
    return resp.text.strip()


def hyde(query: str, provider: str = "ollama", model: str | None = None, tracker: Any | None = None) -> str:
    """Return a hypothetical answer paragraph to embed instead of the query."""
    return _chat(_HYDE.format(query=query), provider, model, tracker)


def step_back(query: str, provider: str = "ollama", model: str | None = None, tracker: Any | None = None) -> str:
    """Return a broader background question for the query."""
    return _chat(_STEP_BACK.format(query=query), provider, model, tracker)


def multi_query(
    query: str,
    n: int = 3,
    provider: str = "ollama",
    model: str | None = None,
    tracker: Any | None = None,
) -> list[str]:
    """Return ``n`` paraphrases of the query (plus the original)."""
    text = _chat(_MULTI.format(n=n, query=query), provider, model, tracker)
    variants = [line.strip("-• ").strip() for line in text.splitlines() if line.strip()]
    return [query, *variants[:n]]


def reciprocal_rank_fusion(ranked_lists: list[list[Hit]], k: int = 4, c: int = 60) -> list[Hit]:
    """Fuse several ranked hit lists into one via RRF (score = sum 1/(c+rank))."""
    scores: dict[str, float] = {}
    best: dict[str, Hit] = {}
    for hits in ranked_lists:
        for rank, h in enumerate(hits):
            key = f"{h.chunk.source}#{h.chunk.index}"
            scores[key] = scores.get(key, 0.0) + 1.0 / (c + rank)
            best.setdefault(key, h)
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return [Hit(best[key].chunk, score) for key, score in ranked[:k]]


def rag_fusion(
    query: str,
    retriever: Any,
    n: int = 3,
    k: int = 4,
    provider: str = "ollama",
    model: str | None = None,
    tracker: Any | None = None,
) -> list[Hit]:
    """Multi-query retrieval fused with RRF — the RAG-Fusion pattern."""
    queries = multi_query(query, n=n, provider=provider, model=model, tracker=tracker)
    ranked_lists = [retriever.search(q, k=k) for q in queries]
    return reciprocal_rank_fusion(ranked_lists, k=k)
