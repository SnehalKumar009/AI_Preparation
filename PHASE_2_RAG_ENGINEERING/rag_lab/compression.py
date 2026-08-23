"""Context compression (notebook 12).

Even relevant passages carry filler. Compression trims retrieved text to the
sentences that actually matter before it reaches the generator — lowering cost and
mitigating the lost-in-the-middle effect. Extractive compression (default) keeps
the sentences most similar to the query; the store and pipeline stay unchanged.
"""

from __future__ import annotations

import numpy as np

from rag_lab.chunking import _sentences
from rag_lab.embeddings import embed_texts
from rag_lab.vectorstore import Hit


def compress_hit(
    query: str,
    text: str,
    keep: int = 2,
    provider: str = "ollama",
    model: str | None = None,
) -> str:
    """Keep the ``keep`` sentences most similar to the query (extractive)."""
    sents = _sentences(text)
    if len(sents) <= keep:
        return text
    q = embed_texts([query], provider=provider, model=model)[0]
    vecs = embed_texts(sents, provider=provider, model=model)
    scores = vecs @ q
    keep_idx = sorted(np.argsort(scores)[::-1][:keep])  # restore original order
    return " ".join(sents[i] for i in keep_idx)


def compress_context(
    query: str,
    hits: list[Hit],
    keep: int = 2,
    provider: str = "ollama",
    model: str | None = None,
) -> str:
    """Compress each hit, then join into one [source]-tagged context block."""
    parts = [
        f"[{h.chunk.source}] {compress_hit(query, h.chunk.text, keep=keep, provider=provider, model=model)}"
        for h in hits
    ]
    return "\n\n---\n\n".join(parts)
