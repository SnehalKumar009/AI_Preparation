# AI Agents — Phase 3

Go from **agents-from-scratch → expert**. An agent is an **LLM that can think,
decide, and act by using tools**. This phase builds that up one primitive at a
time: the agent loop, tool use, ReAct, planning, reasoning search, reflection,
self-correction, control, memory, safety, agentic RAG, and light multi-agent
coordination — one topic per notebook — then maps it all onto real frameworks.

It **reuses Phase 1 `study_buddy`** (providers + cost tracking) and **Phase 2
`rag_lab`** (retrieval), and adds a new `agents_lab/` core that grows as you
progress. Every notebook names which client it uses (**Ollama / OpenAI /
Anthropic / DeepSeek**) and falls back to Ollama when a key is missing.

Everything is built **from scratch** so the mechanics stay visible; a single
bridge notebook maps our tools onto the OpenAI Agents SDK and LangGraph. Deep
frameworks and orchestration are **Phase 5**.

## Setup (.venv, no Docker)

```powershell
# from the AGENTIC_AI root, reuse the existing venv
.\.venv\Scripts\Activate.ps1

# install the Phase 3 superset (also covers Phase 1 & 2 deps)
pip install -r PHASE_3_AI_AGENTS\requirements.txt

# local models (optional but recommended — every notebook can fall back to Ollama)
ollama pull nomic-embed-text
ollama pull qwen3-coder:30b
```

Cloud keys live in `PHASE_1_LLM_FUNDAMENTALS/.env` (reused). Missing keys skip cleanly.

## How to run

Open one notebook at a time and run top-to-bottom:

```powershell
jupyter lab PHASE_3_AI_AGENTS\notebooks
```

## Curriculum

Work through `notebooks/01_*.ipynb` → `19_*.ipynb`, then `capstone.ipynb`.

### Foundations
| # | Topic | # | Topic |
|---|-------|---|-------|
| 01 | The Agent Loop | 03 | Tool Design & Registry |
| 02 | Tool Calling in a Loop | 04 | ReAct (Reason + Act) |

### Reasoning
| # | Topic | # | Topic |
|---|-------|---|-------|
| 05 | Chain-of-Thought & Self-Consistency | 07 | Plan-and-Execute vs ReWOO |
| 06 | Planning — Task Decomposition | 08 | Tree of Thoughts |

### Self-improvement
| # | Topic | # | Topic |
|---|-------|---|-------|
| 09 | Reflection / Self-Critique | 11 | Self-Correction, Retry & JSON Repair |
| 10 | Reflexion (memory across attempts) | — | — |

### Control & safety
| # | Topic | # | Topic |
|---|-------|---|-------|
| 12 | Loop Control & Budget Guards | 14 | Agent Safety — Injection & PII |
| 13 | Short-Term / Working Memory | 15 | Agentic RAG (retrieval as a tool) |

### Multi-agent & bridge
| # | Topic | # | Topic |
|---|-------|---|-------|
| 16 | Supervisor + Workers | 18 | Tracing an Agent |
| 17 | Handoffs & Debate | 19 | Framework Bridge (OpenAI SDK / LangGraph) |

**Capstone:** a from-scratch research agent — tools + agentic RAG + reflection,
with a live USD cost meter.

## Layout

```
PHASE_3_AI_AGENTS/
  agents_lab/      # Phase 3 core (reuses study_buddy + rag_lab)
  notebooks/       # 01..19 + capstone
  requirements.txt # superset of Phases 1 & 2
```

## Provider mix

| Role | Providers used |
|------|----------------|
| Agent loop / reasoning | Ollama (default), OpenAI, Anthropic, DeepSeek (swap one string) |
| Embeddings (agentic RAG) | Ollama `nomic-embed-text` (reused from Phase 2) |
| Framework bridge (19) | OpenAI Agents SDK, LangGraph (optional installs) |

Only have Ollama? Every notebook falls back to local models; cloud cells skip
cleanly when a key is missing.

## Reuse from earlier phases

- **Phase 1 `study_buddy`** — `get_provider`, `chat(..., tools=...)`, `CostTracker`, example tools.
- **Phase 2 `rag_lab`** — `NumpyStore` / `load_markdown` (agentic RAG), `guardrails` (agent safety).

## Deferred to later phases

MCP → Phase 4 · deep orchestration & frameworks → Phase 5 · memory taxonomy →
Phase 6 · evaluation → Phase 7 · production/observability/security → Phase 8 ·
architecture → Phase 9 · real systems → Phase 10.
