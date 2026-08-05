"""rag_lab: the Phase 2 RAG core.

It reuses the Phase 1 ``study_buddy`` package for providers (chat + embeddings)
and cost tracking, and adds retrieval machinery on top. Importing ``rag_lab``
locates the sibling ``LLM_FUNDAMENTALS/study_buddy`` and puts it on ``sys.path``
automatically, so notebooks only need to add ``RAG_ENGINEERING`` to the path.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

# ---- Locate and expose the Phase 1 study_buddy package --------------------
_PKG_DIR = _Path(__file__).resolve().parent          # .../RAG_ENGINEERING/rag_lab
_PHASE2_ROOT = _PKG_DIR.parent                        # .../RAG_ENGINEERING


def _bootstrap_study_buddy() -> _Path:
    """Add sibling LLM_FUNDAMENTALS (which holds study_buddy) to sys.path."""
    for candidate in [_PHASE2_ROOT.parent, *_PHASE2_ROOT.parents]:
        phase1 = candidate / "LLM_FUNDAMENTALS"
        if (phase1 / "study_buddy").exists():
            if str(phase1) not in _sys.path:
                _sys.path.insert(0, str(phase1))
            return phase1
    raise ImportError(
        "Could not locate LLM_FUNDAMENTALS/study_buddy next to RAG_ENGINEERING."
    )


PHASE1_ROOT = _bootstrap_study_buddy()

#: Default corpus directory for Phase 2 notebooks.
NOTES_DIR = _PHASE2_ROOT / "notes"

from rag_lab.embeddings import cosine, embed_texts, l2_normalize
from rag_lab.chunking import Chunk, fixed_chunks, recursive_chunks, sentence_chunks
from rag_lab.vectorstore import NumpyStore, ChromaStore, Hit
from rag_lab.corpus import load_markdown

__all__ = [
    "PHASE1_ROOT",
    "NOTES_DIR",
    "cosine",
    "embed_texts",
    "l2_normalize",
    "Chunk",
    "fixed_chunks",
    "recursive_chunks",
    "sentence_chunks",
    "NumpyStore",
    "ChromaStore",
    "Hit",
    "load_markdown",
]
