"""Chunking strategies.

A ``Chunk`` carries text plus metadata (source, position, and any extra fields),
so later notebooks can demonstrate metadata filtering, parent-child retrieval,
and citations without changing the data model.

Strategies here (more are added in their own notebooks):
- ``fixed_chunks``      — word windows with overlap (the classic baseline)
- ``recursive_chunks``  — split on paragraph/sentence boundaries, then pack
- ``sentence_chunks``   — sentence windows with sentence overlap
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Chunk:
    text: str
    source: str = "inline"
    index: int = 0                       # position of this chunk within its source
    meta: dict[str, Any] = field(default_factory=dict)


def fixed_chunks(
    text: str,
    source: str = "inline",
    chunk_size: int = 200,
    overlap: int = 40,
) -> list[Chunk]:
    """Overlapping word windows. Simple, fast, boundary-agnostic."""
    words = text.split()
    if not words:
        return []
    step = max(1, chunk_size - overlap)
    chunks: list[Chunk] = []
    for i in range(0, len(words), step):
        window = words[i : i + chunk_size]
        if window:
            chunks.append(Chunk(" ".join(window), source, len(chunks)))
        if i + chunk_size >= len(words):
            break
    return chunks


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_SPLIT.split(text.strip()) if s.strip()]


def sentence_chunks(
    text: str,
    source: str = "inline",
    max_sentences: int = 5,
    overlap: int = 1,
) -> list[Chunk]:
    """Group whole sentences so chunks never cut mid-sentence."""
    sents = _sentences(text)
    if not sents:
        return []
    step = max(1, max_sentences - overlap)
    chunks: list[Chunk] = []
    for i in range(0, len(sents), step):
        window = sents[i : i + max_sentences]
        if window:
            chunks.append(Chunk(" ".join(window), source, len(chunks)))
        if i + max_sentences >= len(sents):
            break
    return chunks


def recursive_chunks(
    text: str,
    source: str = "inline",
    chunk_size: int = 200,
    overlap: int = 40,
    separators: list[str] | None = None,
) -> list[Chunk]:
    """Prefer natural boundaries: split on the biggest separator that fits,
    then pack pieces up to ``chunk_size`` words (measured by word count)."""
    separators = separators or ["\n\n", "\n", ". ", " "]

    def split(chunk: str, seps: list[str]) -> list[str]:
        if len(chunk.split()) <= chunk_size or not seps:
            return [chunk]
        sep, rest = seps[0], seps[1:]
        parts = chunk.split(sep) if sep in chunk else [chunk]
        out: list[str] = []
        for p in parts:
            out.extend(split(p, rest) if len(p.split()) > chunk_size else [p])
        return out

    pieces = [p for p in split(text, separators) if p.strip()]

    # Pack small pieces together up to chunk_size words, with word-overlap.
    chunks: list[Chunk] = []
    buf: list[str] = []
    for piece in pieces:
        buf.append(piece.strip())
        if len(" ".join(buf).split()) >= chunk_size:
            words = " ".join(buf).split()
            chunks.append(Chunk(" ".join(words), source, len(chunks)))
            carry = words[-overlap:] if overlap else []
            buf = [" ".join(carry)] if carry else []
    tail = " ".join(buf).strip()
    if tail:
        chunks.append(Chunk(tail, source, len(chunks)))
    return chunks


def parent_child_chunks(
    text: str,
    source: str = "inline",
    parent_size: int = 200,
    child_size: int = 60,
    overlap: int = 10,
) -> tuple[list[Chunk], list[Chunk]]:
    """Split into large parents, then each parent into small children (notebook 13).

    Children carry ``meta['parent_index']`` so a retriever can match on the precise
    child but return the richer parent passage.
    """
    parents = fixed_chunks(text, source=source, chunk_size=parent_size, overlap=0)
    children: list[Chunk] = []
    for parent in parents:
        for child in fixed_chunks(
            parent.text, source=source, chunk_size=child_size, overlap=overlap
        ):
            children.append(
                Chunk(
                    text=child.text,
                    source=source,
                    index=len(children),
                    meta={"parent_index": parent.index},
                )
            )
    return parents, children
