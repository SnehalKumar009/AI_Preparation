"""Contextual retrieval (notebook 22).

A chunk ripped out of its document loses context: "It reduces latency" — what does
"it" refer to? Contextual retrieval (Anthropic, 2024) fixes this by prepending a
short, LLM-generated blurb that situates each chunk inside its parent document
before embedding. The chunk now carries enough context to be retrieved on its own,
which measurably cuts failed retrievals. Context generation is one call per chunk,
so it defaults to Ollama for large corpora.
"""

from __future__ import annotations

from typing import Any

from study_buddy import get_provider

from rag_lab.chunking import Chunk

_CONTEXT = (
    "Here is a document:\n<document>\n{document}\n</document>\n\n"
    "Here is a chunk from it:\n<chunk>\n{chunk}\n</chunk>\n\n"
    "Write a short sentence (max 25 words) situating this chunk within the document "
    "so it can be understood on its own. Reply with only that sentence."
)


def _situate(document: str, chunk: str, llm: Any, model: str | None, tracker: Any | None) -> str:
    resp = llm.chat(
        [{"role": "user", "content": _CONTEXT.format(document=document[:4000], chunk=chunk)}],
        model=model,
        temperature=0.0,
    )
    if tracker is not None:
        tracker.add(resp)
    return resp.text.strip()


def contextualize_chunks(
    chunks: list[Chunk],
    document: str,
    provider: str = "ollama",
    model: str | None = None,
    tracker: Any | None = None,
) -> list[Chunk]:
    """Return new chunks whose text is prefixed with an LLM-written context line.

    The original text is kept in ``meta['original']`` so citations can show the raw
    passage while retrieval benefits from the added context.
    """
    llm = get_provider(provider)
    out: list[Chunk] = []
    for c in chunks:
        prefix = _situate(document, c.text, llm, model, tracker)
        out.append(
            Chunk(
                text=f"{prefix}\n\n{c.text}",
                source=c.source,
                index=c.index,
                meta={**c.meta, "context": prefix, "original": c.text},
            )
        )
    return out
