"""End-to-end RAG: retrieve -> (rerank) -> (compress) -> generate (notebook 12+).

A thin orchestrator the later notebooks and the capstone reuse. It takes any
retriever exposing ``search(query, k)`` and any study_buddy provider, and returns
the grounded answer alongside the context it used.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from study_buddy import get_provider

from rag_lab.vectorstore import Hit

SYSTEM = (
    "You are a helpful assistant. Answer the question using ONLY the context below. "
    "If the context does not contain the answer, say you don't know. Cite sources "
    "with their [tag]."
)


@dataclass
class RagResult:
    answer: str
    hits: list[Hit] = field(default_factory=list)
    context: str = ""


def format_context(hits: list[Hit]) -> str:
    """Join hits into a [source]-tagged block for prompt injection."""
    return "\n\n---\n\n".join(f"[{h.chunk.source}] {h.chunk.text}" for h in hits)


def rag_answer(
    query: str,
    retriever: Any,
    provider: str = "ollama",
    model: str | None = None,
    k: int = 4,
    context: str | None = None,
    tracker: Any | None = None,
) -> RagResult:
    """Retrieve context (unless one is supplied) and answer the query with it."""
    hits = retriever.search(query, k=k)
    ctx = context if context is not None else format_context(hits)
    resp = get_provider(provider).chat(
        [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Context:\n{ctx}\n\nQuestion: {query}"},
        ],
        model=model,
        temperature=0.0,
    )
    if tracker is not None:
        tracker.add(resp)
    return RagResult(answer=resp.text, hits=hits, context=ctx)
