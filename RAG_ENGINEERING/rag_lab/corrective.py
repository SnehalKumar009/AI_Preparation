"""Corrective / Self-RAG (notebook 25).

Naive RAG trusts whatever the retriever returns. Corrective RAG (CRAG) adds a
feedback loop: an LLM grades each retrieved chunk, and if too few are relevant it
rewrites the query and retrieves again before answering. This is a lightweight cousin
of Self-RAG's reflection tokens — the model critiques its own evidence and acts on it,
which rescues weak first-pass retrievals. The loop makes several calls, so it defaults
to Ollama.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from study_buddy import get_provider

from rag_lab.pipeline import SYSTEM, format_context
from rag_lab.vectorstore import Hit

_GRADE = (
    "Is the DOCUMENT relevant to answering the QUESTION? Reply with only YES or NO.\n\n"
    "QUESTION: {question}\n\nDOCUMENT: {document}"
)
_REWRITE = (
    "The search for the QUESTION returned poor results. Rewrite it into a better "
    "search query. Reply with only the rewritten query.\n\nQUESTION: {question}"
)


@dataclass
class CorrectiveTrace:
    queries: list[str] = field(default_factory=list)
    kept: list[Hit] = field(default_factory=list)
    rewrites: int = 0


def _grade(llm: Any, question: str, doc: str, model: str | None, tracker: Any | None) -> bool:
    resp = llm.chat(
        [{"role": "user", "content": _GRADE.format(question=question, document=doc)}],
        model=model,
        temperature=0.0,
    )
    if tracker is not None:
        tracker.add(resp)
    return resp.text.strip().upper().startswith("Y")


def corrective_answer(
    question: str,
    retriever: Any,
    provider: str = "ollama",
    model: str | None = None,
    k: int = 4,
    min_relevant: int = 2,
    max_rewrites: int = 1,
    tracker: Any | None = None,
) -> tuple[str, CorrectiveTrace]:
    """Grade retrieved docs; rewrite+retry if too few are relevant, then answer."""
    llm = get_provider(provider)
    trace = CorrectiveTrace()
    query = question

    for attempt in range(max_rewrites + 1):
        trace.queries.append(query)
        hits = retriever.search(query, k=k)
        relevant = [h for h in hits if _grade(llm, question, h.chunk.text, model, tracker)]
        if len(relevant) >= min_relevant or attempt == max_rewrites:
            trace.kept = relevant or hits
            break
        # Not enough good evidence — rewrite the query and try once more.
        resp = llm.chat(
            [{"role": "user", "content": _REWRITE.format(question=question)}],
            model=model,
            temperature=0.3,
        )
        if tracker is not None:
            tracker.add(resp)
        query = resp.text.strip().splitlines()[0]
        trace.rewrites += 1

    resp = llm.chat(
        [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Context:\n{format_context(trace.kept)}\n\nQuestion: {question}"},
        ],
        model=model,
        temperature=0.0,
    )
    if tracker is not None:
        tracker.add(resp)
    return resp.text, trace
