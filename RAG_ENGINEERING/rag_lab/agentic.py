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
    'If the answer requires multiple facts to be known, provide one query per json to get the facts.'
    "Base every answer only on retrieved context and cite [source] tags."
)


@dataclass
class AgentTrace:
    searches: list[str] = field(default_factory=list)
    hits: list[Hit] = field(default_factory=list)


def _parse_json(text: str) -> dict[str, Any]:
    print(f"DEBUG _parse_json: Parsing text - {text}")
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        print("DEBUG _parse_json: No JSON found in response")
        return {}
    try:
        result = json.loads(m.group())
        print(f"DEBUG _parse_json: Successfully parsed JSON - {result}")
        return result
    except json.JSONDecodeError as e:
        print(f"DEBUG _parse_json: JSON decode error - {e}")
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
    print(f"DEBUG agentic_answer: Starting with question - {question}")
    
    llm = get_provider(provider)
    print(f"DEBUG agentic_answer: Initialized LLM provider - {provider}")
    
    trace = AgentTrace()
    print(" agentic_answer: Initialized empty trace")
    
    scratch = ""
    print("DEBUG agentic_answer: Starting decision loop")
    
    for step in range(max_steps):
        print(f"DEBUG agentic_answer: Step {step + 1} of {max_steps}")
        
        user = f"QUESTION: {question}\n\nRETRIEVED SO FAR:\n{scratch or '(nothing yet)'}"
        print(f"DEBUG agentic_answer: Preparing user prompt for step {step + 1}")
        print(f"DEBUG agentic_answer: User prompt content - {user[:20000]}...")
        
        resp = llm.chat(
            [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": user}],
            model=model,
            temperature=0.0,
        )
        print(f"DEBUG agentic_answer: LLM response received - {resp.text[:10000]}...")
        
        if tracker is not None:
            print("DEBUG agentic_answer: Adding response to tracker")
            tracker.add(resp)
        
        decision = _parse_json(resp.text)
        print(f"DEBUG agenDEBUGtic_answer: Parsed decision - {decision}")
        
        if decision.get("action") == "answer":
            print("DEBUG agentic_answer: Model decided to answer")
            answer = decision.get("answer", resp.text)
            print(f"DEBUG agentic_answer: Final answer - {answer[:10000]}...")
            return answer, trace
        
        query = decision.get("query") or question
        print(f"DEBUG agentic_answer: Selected query for search - {query}")
        
        trace.searches.append(query)
        print(f"DEBUG agentic_answer: Added query to trace searches")
        
        hits = retriever.search(query, k=k)
        print(f"DEBUG agentic_answer: Retrieved {len(hits)} hits for query '{query}'")
        
        trace.hits.extend(hits)
        print(f"DEBUG agentic_answer: Extended trace hits with new results")
        
        scratch += ("\n\n" if scratch else "") + format_context(hits)
        print(f"DEBUG agentic_answer: Updated scratch context with new hits")

    print("DEBUG agentic_answer: Maximum steps reached, generating final answer")
    
    resp = llm.chat(
        [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Context:\n{scratch}\n\nQuestion: {question}"},
        ],
        model=model,
        temperature=0.0,
    )
    print(f"DEBUG agentic_answer: Final answer generation complete - {resp.text[:10000]}...")
    
    if tracker is not None:
        print("DEBUG agentic_answer: Adding final response to tracker")
        tracker.add(resp)
        
    return resp.text, trace
