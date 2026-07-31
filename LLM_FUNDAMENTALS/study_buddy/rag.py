"""Minimal RAG: chunk -> embed -> store (NumPy) -> retrieve by cosine similarity.

Kept deliberately dependency-light (just NumPy) so the mechanics of retrieval
are visible for learning. Embeddings default to local Ollama nomic-embed-text.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from study_buddy.config import settings
from study_buddy.providers import get_provider
from study_buddy.providers.base import LLMProvider


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Split text into overlapping word windows."""
    words = text.split()
    if not words:
        return []
    step = max(1, chunk_size - overlap)
    chunks = []
    for i in range(0, len(words), step):
        window = words[i : i + chunk_size]
        if window:
            chunks.append(" ".join(window))
        if i + chunk_size >= len(words):
            break
    return chunks


@dataclass
class Retrieved:
    text: str
    score: float
    source: str


class RagStore:
    """In-memory vector store backed by a NumPy matrix."""

    def __init__(self, embed_provider: LLMProvider | None = None, embed_model: str | None = None):
        self.provider = embed_provider or get_provider("ollama")
        self.embed_model = embed_model or settings.ollama_embed_model
        self.chunks: list[str] = []
        self.sources: list[str] = []
        self._matrix: np.ndarray | None = None

    @staticmethod
    def _normalize(vecs: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vecs / norms

    def add_texts(self, texts: list[str], source: str = "inline") -> None:
        all_chunks: list[str] = []
        for t in texts:
            all_chunks.extend(chunk_text(t))
        if not all_chunks:
            return
        vectors = np.array(self.provider.embed(all_chunks, model=self.embed_model), dtype=np.float32)
        vectors = self._normalize(vectors)
        self.chunks.extend(all_chunks)
        self.sources.extend([source] * len(all_chunks))
        self._matrix = vectors if self._matrix is None else np.vstack([self._matrix, vectors])

    def add_directory(self, path: str | Path | None = None, pattern: str = "*.md") -> int:
        """Index every matching file in a directory. Returns file count."""
        path = Path(path) if path else settings.notes_dir
        files = sorted(path.glob(pattern)) + sorted(path.glob("*.txt"))
        for f in files:
            self.add_texts([f.read_text(encoding="utf-8")], source=f.name)
        return len(files)

    def search(self, query: str, k: int = 4) -> list[Retrieved]:
        if self._matrix is None:
            return []
        q = np.array(self.provider.embed([query], model=self.embed_model), dtype=np.float32)
        q = self._normalize(q)[0]
        scores = self._matrix @ q  # cosine similarity (all vectors normalized)
        top = np.argsort(scores)[::-1][:k]
        return [Retrieved(self.chunks[i], float(scores[i]), self.sources[i]) for i in top]

    def context_for(self, query: str, k: int = 4) -> str:
        """Concatenated retrieved chunks, ready to inject into a prompt."""
        hits = self.search(query, k=k)
        return "\n\n---\n\n".join(f"[{h.source}] {h.text}" for h in hits)
