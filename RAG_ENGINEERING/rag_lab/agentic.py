"""Agentic RAG: let the model decide when to retrieve (notebook 15).

Instead of always retrieving, the model runs a small decision loop: at each step it
replies with a JSON action — ``search`` for a query, or ``answer`` when it has
enough. This ReAct-style loop is portable across providers (no native tool-calling
required) and handles queries that need no retrieval or several targeted lookups.
The loop can make several calls, so it defaults to Ollama.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from study_buddy import get_provider

from rag_lab.pipeline import SYSTEM, format_context
from rag_lab.vectorstore import Hit

_SYSTEM = (
    "You are a retrieval agent. Each step, reply with ONE JSON object and nothing "
    'else:\n  {"action": "search", "query": "..."}  to look something up, or\n'
    '  {"action": "answer", "answer": "..."}  when you can answer.\n'
    "Base every answer only on retrieved context and cite [source] tags."
)


@dataclass
class AgentTrace:
    searches: list[str] = field(default_factory=list)
    hits: list[Hit] = field(default_factory=list)


def _parse_json(text: str) -> dict[str, Any]:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return {}
    try:
        return json.loads(m.group())
    except json.JSONDecodeError:
        return {}


def agentic_answer(
    question: str,
    retriever: Any,
    provider: str = "ollama",
    model: str | None = None,
    k: int = 3,
    max_steps: int = 4,
    tracker: Any | None = None,
) -> tuple[str, AgentTrace]:
    """Run a decision loop where the model chooses when and what to retrieve."""
    llm = get_provider(provider)
    trace = AgentTrace()
    scratch = ""
    for _ in range(max_steps):
        user = f"QUESTION: {question}\n\nRETRIEVED SO FAR:\n{scratch or '(nothing yet)'}"
        resp = llm.chat(
            [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": user}],
            model=model,
            temperature=0.0,
        )
        if tracker is not None:
            tracker.add(resp)
        decision = _parse_json(resp.text)
        if decision.get("action") == "answer":
            return decision.get("answer", resp.text), trace
        query = decision.get("query") or question
        trace.searches.append(query)
        hits = retriever.search(query, k=k)
        trace.hits.extend(hits)
        scratch += ("\n\n" if scratch else "") + format_context(hits)

    resp = llm.chat(
        [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Context:\n{scratch}\n\nQuestion: {question}"},
        ],
        model=model,
        temperature=0.0,
    )
    if tracker is not None:
        tracker.add(resp)
    return resp.text, trace
