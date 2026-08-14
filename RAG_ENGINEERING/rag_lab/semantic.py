"""Semantic chunking (notebook 19).

Fixed-size chunking cuts text on arbitrary word counts, often splitting a single
idea across two chunks. Semantic chunking instead places boundaries where the
*meaning* shifts: embed each sentence, walk them in order, and start a new chunk
whenever the next sentence is dissimilar enough from the current chunk's running
centroid. The result is topically coherent passages that embed and retrieve better.
"""

from __future__ import annotations

import numpy as np

from rag_lab.chunking import Chunk, _sentences
from rag_lab.embeddings import embed_texts


def semantic_chunks(
    text: str,
    source: str = "inline",
    threshold: float = 0.5,
    max_sentences: int = 8,
    provider: str = "ollama",
    model: str | None = None,
) -> list[Chunk]:
    """Group consecutive sentences until the topic drifts below ``threshold``.

    A boundary is placed when a sentence's cosine similarity to the current
    chunk's mean embedding drops under ``threshold`` (or the chunk hits
    ``max_sentences``). Lower ``threshold`` => fewer, larger chunks.
    """
    sents = _sentences(text)
    if not sents:
        return []
    vecs = embed_texts(sents, provider=provider, model=model)

    chunks: list[Chunk] = []
    cur: list[int] = [0]
    for i in range(1, len(sents)):
        centroid = vecs[cur].mean(axis=0)
        sim = float(centroid @ vecs[i] / ((np.linalg.norm(centroid) or 1.0)))
        if sim < threshold or len(cur) >= max_sentences:
            chunks.append(Chunk(" ".join(sents[j] for j in cur), source, len(chunks)))
            cur = [i]
        else:
            cur.append(i)
    if cur:
        chunks.append(Chunk(" ".join(sents[j] for j in cur), source, len(chunks)))
    return chunks
