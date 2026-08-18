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
    "You are researching to answer a multi-part QUESTION by chaining facts across "
    "documents.\n\n"
    "Step 1: List the specific named facts the QUESTION requires (people, companies, "
    "dates, etc.).\n"
    "Step 2: Check CONTEXT SO FAR against that list. Mark each fact FOUND only if its "
    "exact value is explicitly written in the context -- never mark something FOUND "
    "because you happen to know it from outside knowledge; that counts as MISSING.\n"
    "Step 3: If anything is MISSING, decide the ONE search query most likely to "
    "retrieve it -- name the specific missing entity or relationship, don't just "
    "repeat the QUESTION verbatim. If every fact has an explicit, concrete value "
    "written in the context, you're done.\n\n"
    "{done_rule}\n\n"
    "QUESTION: {question}\n\n"
    "QUERIES ALREADY TRIED: {tried}\n\n"
    "CONTEXT SO FAR:\n{context}\n\n"
    "Show your Step 1/Step 2 reasoning briefly, then end with exactly one final "
    "line:\nNEXT: <search query>\nor\nDONE"
)

_DONE_RULE_NORMAL = (
    "Do not answer DONE just because the context is long or topically related -- "
    "only when every required fact has an explicit value."
)
_DONE_RULE_MIN_HOPS = (
    "This is search {hop} of at least {min_hops} required searches for a question "
    "this complex -- you may NOT answer DONE yet, even if the context looks "
    "sufficient. Propose the next query."
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
    min_hops: int = 2,
    tracker: Any | None = None,
) -> tuple[str, HopTrace]:
    """Iteratively retrieve, letting an LLM steer each next query, then answer.

    ``min_hops`` is a hard backstop: the loop will not honor a DONE verdict
    before that many hops have actually run, even if the model claims the
    context is already sufficient. Small/local models in particular tend to
    call DONE after a single retrieval pass.
    """
    llm = get_provider(provider)
    trace = HopTrace()
    query = question
    for hop_index in range(max_hops):
        trace.queries.append(query)
        trace.hits.extend(retriever.search(query, k=k))

        must_continue = hop_index + 1 < min_hops
        done_rule = (
            _DONE_RULE_MIN_HOPS.format(hop=hop_index + 1, min_hops=min_hops)
            if must_continue
            else _DONE_RULE_NORMAL
        )
        context1 = _FOLLOWUP.format(
                        question=question,
                        tried=", ".join(trace.queries) or "(none yet)",
                        context=format_context(trace.hits),
                        done_rule=done_rule,
                    )
        print("----------------------------------------------------")
        print(context1)
        print("----------------------------------------------------")
        resp = llm.chat(
            [
                {
                    "role": "user",
                    "content": _FOLLOWUP.format(
                        question=question,
                        tried=", ".join(trace.queries) or "(none yet)",
                        context=format_context(trace.hits),
                        done_rule=done_rule,
                    ),
                }
            ],
            model=model,
            temperature=0.0,
        )
        if tracker is not None:
            tracker.add(resp)

        lines = [line.strip() for line in resp.text.strip().splitlines() if line.strip()]
        last = lines[-1] if lines else ""
        if not last:
            break
        if last.upper() == "DONE":
            if must_continue:
                # Model ignored the instruction not to stop yet. Don't trust
                # it -- force another hop off the original question rather
                # than exiting after a single retrieval pass.
                query = question
                continue
            break
        query = last[5:].strip() if last.upper().startswith("NEXT:") else last

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
