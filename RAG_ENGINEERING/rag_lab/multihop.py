"""Multi-hop retrieval (notebook 14).

Some questions need a chain of facts: answer a sub-question, notice what's still
missing, retrieve again. This loops retrieval plus an LLM that proposes the next
query, accumulating context until it can answer (or a hop cap is hit). The loop can
run many small chat calls, so it defaults to Ollama.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from study_buddy import get_provider

from rag_lab.pipeline import SYSTEM, format_context
from rag_lab.vectorstore import Hit

_FOLLOWUP = (
    "You are researching to answer a QUESTION. Given what you have retrieved so far, "
    "either output the single next search query that would fill the biggest gap, or "
    "output DONE if the context is already sufficient. Reply with only the query or "
    "DONE.\n\nQUESTION: {question}\n\nCONTEXT SO FAR:\n{context}\n\nNext query or DONE:"
)


@dataclass
class HopTrace:
    queries: list[str] = field(default_factory=list)
    hits: list[Hit] = field(default_factory=list)


def multihop_answer(
    question: str,
    retriever: Any,
    provider: str = "ollama",
    model: str | None = None,
    k: int = 3,
    max_hops: int = 3,
    tracker: Any | None = None,
) -> tuple[str, HopTrace]:
    """Iteratively retrieve, letting an LLM steer each next query, then answer."""
    llm = get_provider(provider)
    trace = HopTrace()
    query = question
    for _ in range(max_hops):
        trace.queries.append(query)
        trace.hits.extend(retriever.search(query, k=k))
        resp = llm.chat(
            [
                {
                    "role": "user",
                    "content": _FOLLOWUP.format(
                        question=question, context=format_context(trace.hits)
                    ),
                }
            ],
            model=model,
            temperature=0.0,
        )
        if tracker is not None:
            tracker.add(resp)
        nxt = resp.text.strip()
        if not nxt or nxt.upper().startswith("DONE"):
            break
        query = nxt.splitlines()[0].strip()

    resp = llm.chat(
        [
            {"role": "system", "content": SYSTEM},
            {
                "role": "user",
                "content": f"Context:\n{format_context(trace.hits)}\n\nQuestion: {question}",
            },
        ],
        model=model,
        temperature=0.0,
    )
    if tracker is not None:
        tracker.add(resp)
    return resp.text, trace
