# RAG Engineering — Phase 2

Go from **RAG-from-scratch → expert**. This phase builds a deep, hands-on
understanding of Retrieval-Augmented Generation: chunking, embeddings, vector
search, hybrid retrieval, reranking, compression, agentic/graph RAG, evaluation,
and production concerns — one topic per notebook.

It **reuses the Phase 1 `study_buddy` package** (providers, cost tracking) and
adds a new `rag_lab/` core that grows as you progress. Every notebook names
which client it uses (**Ollama / OpenAI / Anthropic / DeepSeek**) so you learn
each SDK, not just one.

## Setup (.venv, no Docker)

```powershell
# from the AGENTIC_AI root, reuse the existing venv
.\.venv\Scripts\Activate.ps1

# install the Phase 2 superset (also covers Phase 1 deps)
pip install -r RAG_ENGINEERING\requirements.txt

# local models (optional but recommended — every notebook can fall back to Ollama)
ollama pull nomic-embed-text
ollama pull qwen3-coder:30b
```

Cloud keys live in `LLM_FUNDAMENTALS/.env` (reused). Missing keys skip cleanly.

## How to run

Open one notebook at a time and run top-to-bottom:

```powershell
jupyter lab RAG_ENGINEERING\notebooks
```

## Curriculum

Work through `notebooks/01_*.ipynb` → `29_*.ipynb`, then `capstone.ipynb`.

### Core retrieval
| # | Topic | # | Topic |
|---|-------|---|-------|
| 01 | Embeddings | 05 | Metadata |
| 02 | Chunking | 06 | Cosine Similarity |
| 03 | Chunk Size | 07 | Similarity Search |
| 04 | Overlap | 08 | Vector Databases |

### Advanced retrieval
| # | Topic | # | Topic |
|---|-------|---|-------|
| 09 | Keyword Search (BM25) | 11 | Reranking |
| 10 | Hybrid Search | 12 | Context Compression |

### Frontier RAG
| # | Topic | # | Topic |
|---|-------|---|-------|
| 13 | Parent-Child Chunking | 16 | Graph RAG |
| 14 | Multi-hop Retrieval | 17 | Citation Generation |
| 15 | Agentic RAG | — | — |

### Expert block
| # | Topic | # | Topic |
|---|-------|---|-------|
| 18 | Document Ingestion & Parsing | 24 | ANN Indexing Internals |
| 19 | Semantic Chunking | 25 | Corrective / Self-RAG |
| 20 | Query Transformation | 26 | Security & Guardrails |
| 21 | RAG Evaluation | 27 | Semantic Caching |
| 22 | Contextual Retrieval | 28 | Long-Context vs RAG |
| 23 | Sentence-Window Retrieval | 29 | Multimodal RAG |

**Capstone:** hybrid + rerank + compression + citations app with an evaluation
harness and a live USD cost meter.

## Layout

```
RAG_ENGINEERING/
  rag_lab/         # Phase 2 core (reuses study_buddy providers)
  notebooks/       # 01..29 + capstone
  notes/           # RAG corpus (multi-section markdown)
  webui/           # Streamlit capstone
```

## Provider mix

| Role | Providers used |
|------|----------------|
| Embeddings | Ollama `nomic-embed-text`, OpenAI `text-embedding-3-small` |
| Chat / rerank / eval | Anthropic, OpenAI, DeepSeek, Ollama (rotated) |
| Vision (29) | Anthropic, Gemini |

Only have Ollama? Every notebook falls back to local models; cloud cells skip
cleanly when a key is missing.
