"""Long-context vs RAG (notebook 28).

Modern models have huge context windows, tempting you to skip retrieval and just
"stuff" the whole corpus into the prompt. This module helps compare the two
approaches head-to-head: ``stuff_answer`` dumps everything in, ``rag_answer``
retrieves first. The trade-off is cost and focus — stuffing pays for every token
every call and dilutes attention (lost-in-the-middle), while RAG sends only what
matters but can miss a passage the retriever ranked low.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from study_buddy import get_provider, count_tokens

from rag_lab.chunking import Chunk
from rag_lab.pipeline import SYSTEM


@dataclass
class ApproachResult:
    answer: str
    prompt_tokens: int
    approach: str


def stuff_context(chunks: list[Chunk]) -> str:
    """Concatenate the entire corpus into one [source]-tagged block."""
    return "\n\n---\n\n".join(f"[{c.source}] {c.text}" for c in chunks)


def stuff_answer(
    query: str,
    chunks: list[Chunk],
    provider: str = "ollama",
    model: str | None = None,
    tracker: Any | None = None,
) -> ApproachResult:
    """Answer by putting the whole corpus in the prompt (no retrieval)."""
    context = stuff_context(chunks)
    resp = get_provider(provider).chat(
        [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
        ],
        model=model,
        temperature=0.0,
    )
    if tracker is not None:
        tracker.add(resp)
    return ApproachResult(resp.text, count_tokens(context), "long-context")
