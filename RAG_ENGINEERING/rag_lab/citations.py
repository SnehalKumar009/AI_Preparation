"""Citation generation and grounding checks (notebook 17).

Grounded generation answers only from retrieved context and tags each sentence with
its ``[source]``. A simple verifier then checks that every cited tag was actually
retrieved — an uncited or mis-cited claim is a hallucination signal worth flagging.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from study_buddy import get_provider

from rag_lab.vectorstore import Hit

_CITE = re.compile(r"\[([^\]]+)\]")

_SYSTEM = (
    "Answer the QUESTION using only the CONTEXT. After each sentence, cite the source "
    "it came from in square brackets, e.g. [rag_overview.md]. Use only the source tags "
    "shown in the context. If the context does not support an answer, say you don't know."
)


@dataclass
class CitedAnswer:
    answer: str
    cited: list[str] = field(default_factory=list)
    valid: list[str] = field(default_factory=list)
    invalid: list[str] = field(default_factory=list)


def cited_answer(
    question: str,
    hits: list[Hit],
    provider: str = "ollama",
    model: str | None = None,
    tracker: Any | None = None,
) -> CitedAnswer:
    """Generate an answer with inline citations and validate them against sources."""
    context = "\n\n---\n\n".join(f"[{h.chunk.source}] {h.chunk.text}" for h in hits)
    print("===================================================")
    print(context)
    print("===================================================")
    resp = get_provider(provider).chat(
        [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": f"CONTEXT:\n{context}\n\nQUESTION: {question}"},
        ],
        model=model,
        temperature=0.0,
    )
    print("===================================================")
    print(resp)
    print("===================================================")
    if tracker is not None:
        tracker.add(resp)
    sources = {h.chunk.source for h in hits}
    cited = list(dict.fromkeys(_CITE.findall(resp.text)))
    valid = [c for c in cited if c in sources]
    invalid = [c for c in cited if c not in sources]
    return CitedAnswer(answer=resp.text, cited=cited, valid=valid, invalid=invalid)
