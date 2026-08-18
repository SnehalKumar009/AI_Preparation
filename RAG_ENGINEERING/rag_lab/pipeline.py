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
    "Never use outside or prior knowledge to fill a gap, even if you are confident "
    "it is correct and even if it is well-known or easily verifiable -- a fact you "
    "recall but that is not written in the context below counts as unknown. If any "
    "single fact needed for the answer is missing from the context, say specifically "
    "which fact is missing and that you don't know, rather than completing the chain "
    "yourself. Cite every fact you use with its [tag]."
)


@dataclass
class RagResult:
    answer: str
    hits: list[Hit] = field(default_factory=list)
    context: str = ""


def format_context(hits: list[Hit]) -> str:
    """Join hits into a [source :: heading]-tagged block for prompt injection.

    The heading rides along with the filename so that pronouns and
    possessives inside an isolated chunk (e.g. "His startup was acquired...")
    still resolve to the right entity even when the chunk is read without
    its neighboring chunks.
    """
    return "\n\n---\n\n".join(
        f"[{h.chunk.source} :: {h.chunk.meta.get('heading', '')}] {h.chunk.text}"
        for h in hits
    )


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
