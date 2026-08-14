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
from rag_lab.chunking import (
    Chunk,
    fixed_chunks,
    parent_child_chunks,
    recursive_chunks,
    sentence_chunks,
)
from rag_lab.vectorstore import NumpyStore, ChromaStore, Hit
from rag_lab.corpus import load_markdown
from rag_lab.keyword import BM25Store, tokenize
from rag_lab.hybrid import HybridRetriever
from rag_lab.rerank import Reranker, cross_encoder_available, cross_encoder_rerank
from rag_lab.compression import compress_context, compress_hit
from rag_lab.pipeline import RagResult, format_context, rag_answer
from rag_lab.parent_child import ParentChildRetriever
from rag_lab.multihop import HopTrace, multihop_answer
from rag_lab.agentic import AgentTrace, agentic_answer
from rag_lab.graph import GraphRAG, Triple
from rag_lab.citations import CitedAnswer, cited_answer
from rag_lab.ingestion import (
    clean_text,
    ingest_directory,
    ingest_path,
    parse_document,
    parse_html,
    parse_markdown,
    parse_pdf,
    parse_text,
)
from rag_lab.semantic import semantic_chunks
from rag_lab.query_transform import (
    hyde,
    multi_query,
    rag_fusion,
    reciprocal_rank_fusion,
    step_back,
)
from rag_lab.evaluation import (
    EvalResult,
    Judge,
    answer_relevance,
    context_precision,
    context_recall,
    evaluate,
    faithfulness,
)
from rag_lab.contextual import contextualize_chunks
from rag_lab.sentence_window import SentenceWindowRetriever
from rag_lab.ann import (
    IVFIndex,
    brute_force,
    dequantize,
    hnsw_search,
    quantize,
    recall_at_k,
)
from rag_lab.corrective import CorrectiveTrace, corrective_answer
from rag_lab.guardrails import (
    GuardReport,
    detect_injection,
    is_grounded,
    redact_pii,
    sanitize_hits,
    scan,
)
from rag_lab.semantic_cache import CacheEntry, SemanticCache
from rag_lab.long_context import ApproachResult, stuff_answer, stuff_context
from rag_lab.multimodal import ImageDoc, MultimodalStore, caption_image

__all__ = [
    "PHASE1_ROOT",
    "NOTES_DIR",
    "cosine",
    "embed_texts",
    "l2_normalize",
    "Chunk",
    "fixed_chunks",
    "parent_child_chunks",
    "recursive_chunks",
    "sentence_chunks",
    "NumpyStore",
    "ChromaStore",
    "Hit",
    "load_markdown",
    "BM25Store",
    "tokenize",
    "HybridRetriever",
    "Reranker",
    "cross_encoder_available",
    "cross_encoder_rerank",
    "compress_context",
    "compress_hit",
    "RagResult",
    "format_context",
    "rag_answer",
    "ParentChildRetriever",
    "HopTrace",
    "multihop_answer",
    "AgentTrace",
    "agentic_answer",
    "GraphRAG",
    "Triple",
    "CitedAnswer",
    "cited_answer",
    # 18 ingestion
    "clean_text",
    "ingest_directory",
    "ingest_path",
    "parse_document",
    "parse_html",
    "parse_markdown",
    "parse_pdf",
    "parse_text",
    # 19 semantic chunking
    "semantic_chunks",
    # 20 query transformation
    "hyde",
    "multi_query",
    "rag_fusion",
    "reciprocal_rank_fusion",
    "step_back",
    # 21 evaluation
    "EvalResult",
    "Judge",
    "answer_relevance",
    "context_precision",
    "context_recall",
    "evaluate",
    "faithfulness",
    # 22 contextual retrieval
    "contextualize_chunks",
    # 23 sentence-window
    "SentenceWindowRetriever",
    # 24 ANN internals
    "IVFIndex",
    "brute_force",
    "dequantize",
    "hnsw_search",
    "quantize",
    "recall_at_k",
    # 25 corrective / self-RAG
    "CorrectiveTrace",
    "corrective_answer",
    # 26 guardrails
    "GuardReport",
    "detect_injection",
    "is_grounded",
    "redact_pii",
    "sanitize_hits",
    "scan",
    # 27 semantic caching
    "CacheEntry",
    "SemanticCache",
    # 28 long-context vs RAG
    "ApproachResult",
    "stuff_answer",
    "stuff_context",
    # 29 multimodal
    "ImageDoc",
    "MultimodalStore",
    "caption_image",
]
