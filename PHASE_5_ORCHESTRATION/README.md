# Orchestration & Frameworks — Phase 5

Go from **agent loop → orchestrated systems**. A Phase 3 agent is a `while`
loop: the model decides every next step, and all the work lives in one
process's memory. That's why it can't guarantee order, run steps in parallel,
survive a crash, wait hours for a human, or replay a single step.
**Orchestration** fixes all five by making the structure explicit:
**state + nodes + edges + checkpoints**.

This phase builds that from scratch first (`orch_lab`, Part A), then goes
**equally deep** into five frameworks and one durable-execution engine:
LangGraph, the OpenAI Agents SDK, CrewAI, Microsoft Agent Framework, Google ADK
and Temporal. Every framework is taught through the **same six lenses** and
builds the **same reference task**, so the final comparison is apples to
apples.

It reuses Phase 1 `study_buddy` (providers + cost), Phase 2 `rag_lab`,
Phase 3 `agents_lab` (agent loop, tools, tracing) and Phase 4 `mcp_lab`
(the MCP tool servers the frameworks orchestrate).

## How each notebook teaches

Phase 5 notebooks are written to be studied **without a book**. Each one has:

1. **Why this exists** — the problem, and what breaks without it
2. **How it works** — the mechanics, with a diagram
3. **Annotated code** — runnable, with the output explained
4. **Trade-offs & common mistakes** — when not to use it, what bites in production
5. **Check yourself** — questions with hidden answers

Demos that need a failure to happen *on cue* use `orch_lab.ScriptedLLM` (a
deterministic fake model); each notebook then repeats the idea on a real model.

## Setup — one environment per framework

The frameworks pin conflicting dependency versions. Measured on this build,
CrewAI and ADK need `openai` 2.x while the others use 3.x, and CrewAI pins an
older `pydantic`. So **each framework gets its own environment and Jupyter
kernel**, built with [`uv`](https://docs.astral.sh/uv/):

```bash
cd PHASE_5_ORCHESTRATION
./setup.sh              # build all six environments (~2 GB, a few minutes)
./setup.sh core         # ...or only the ones you need
./setup.sh --check      # import-test what's built
```

| Environment | Jupyter kernel | Used by |
|---|---|---|
| `core` | Phase 5 · core | Part A (from scratch), Part G (Temporal), Part H |
| `langgraph` | Phase 5 · langgraph | Part B + capstone |
| `openai_agents` | Phase 5 · openai_agents | Part C |
| `crewai` | Phase 5 · crewai | Part D |
| `msaf` | Phase 5 · msaf | Part E |
| `adk` | Phase 5 · adk | Part F |

Each environment also installs everything Phases 1–4 need, so the lab packages
import in every kernel. Each notebook names its kernel at the top; if you're on
the wrong one, `orch_lab.requires(...)` prints the fix and framework cells skip.

**Models.** Everything runs on local Ollama by default; cloud keys are read from
`PHASE_1_LLM_FUNDAMENTALS/.env` and missing keys skip cleanly.

```bash
ollama pull qwen3-coder:30b
ollama pull nomic-embed-text
```

**Services (Docker).** Docker is used only for infrastructure, not for Python.
Before notebooks 44–45:

```bash
docker compose up -d temporal      # gRPC :7233, web UI http://localhost:8233
```

## How to run

```bash
jupyter lab PHASE_5_ORCHESTRATION/notebooks
```

Open one notebook at a time, check the kernel named in its first cell, and run
top to bottom.

## Curriculum

Status: ✅ written · ⬜ planned

### Part A — Orchestration from scratch (`orch_lab`, kernel: core)
| # | Topic | |
|---|---|---|
| 01 | Why orchestration — where the agent loop breaks | ✅ |
| 02 | Workflows vs agents — the spectrum (chain → router → orchestrator-workers → autonomous) | ⬜ |
| 03 | State — typed shared state and reducers (append vs overwrite) | ⬜ |
| 04 | A graph engine from scratch — nodes, edges, END | ⬜ |
| 05 | Routing and conditional branching | ⬜ |
| 06 | Parallelism — fan-out / fan-in, map-reduce | ⬜ |
| 07 | Cycles with guards — recursion limits and budgets | ⬜ |
| 08 | Failure handling — retries, backoff, timeouts, fallbacks | ⬜ |
| 09 | Checkpointing and resume (SQLite) | ⬜ |
| 10 | Human-in-the-loop — pause, approve or edit, resume | ⬜ |
| 11 | Time travel — replay and fork from a checkpoint | ⬜ |
| 12 | Streaming graph events | ⬜ |

### The six lenses (used for every framework in Parts B–F)
| Lens | Question it answers |
|---|---|
| 1 · Core model | What are its primitives? Mapped 1:1 onto our engine |
| 2 · Tools & state | How does state flow? How are tools attached? |
| 3 · Multi-agent | Delegation, handoffs, teams |
| 4 · Workflow control | Branching, parallelism, retries, streaming |
| 5 · Persistence & HITL | Checkpoints / sessions, pause and resume |
| 6 · MCP + reference task | Phase 4 MCP servers, then the full reference task |

**Reference task:** a research-and-report workflow over the Phase 4 MCP
servers — parallel research, a written draft, a review loop, and a human
approval gate before anything is saved.

### Part B — LangGraph (kernel: langgraph)
| # | Topic | |
|---|---|---|
| 13 | Core model — `StateGraph`, nodes, edges, compile | ⬜ |
| 14 | Tools & state — reducers, `MessagesState`, `ToolNode` | ⬜ |
| 15 | Multi-agent — supervisor vs swarm / handoffs | ⬜ |
| 16 | Workflow control — `Command`, `Send`, retries, streaming | ⬜ |
| 17 | Persistence & HITL — checkpointers, threads, `interrupt()` | ⬜ |
| 18 | Subgraphs and composition | ⬜ |
| 19 | MCP + reference task | ⬜ |

### Part C — OpenAI Agents SDK (kernel: openai_agents)
| # | Topic | |
|---|---|---|
| 20 | Core model — `Agent`, `Runner`, the run loop | ⬜ |
| 21 | Tools & state — function tools, context, structured output | ⬜ |
| 22 | Multi-agent — handoffs and agents-as-tools | ⬜ |
| 23 | Workflow control — guardrails, streaming, code-driven orchestration | ⬜ |
| 24 | Persistence & HITL — sessions, approvals, tracing | ⬜ |
| 25 | MCP + reference task | ⬜ |

### Part D — CrewAI (kernel: crewai)
| # | Topic | |
|---|---|---|
| 26 | Core model — agents, tasks, crews | ⬜ |
| 27 | Tools & state — tools, context passing, structured output | ⬜ |
| 28 | Multi-agent — sequential vs hierarchical process | ⬜ |
| 29 | Workflow control — Flows (`@start`, `@listen`, `@router`) | ⬜ |
| 30 | Persistence & HITL — Flow state persistence, human input | ⬜ |
| 31 | MCP + reference task | ⬜ |

### Part E — Microsoft Agent Framework (kernel: msaf)
| # | Topic | |
|---|---|---|
| 32 | Core model — agents, threads, and the AutoGen / Semantic Kernel lineage | ⬜ |
| 33 | Tools & state — function tools, middleware | ⬜ |
| 34 | Multi-agent — orchestration patterns | ⬜ |
| 35 | Workflow control — workflows, executors, edges | ⬜ |
| 36 | Persistence & HITL — checkpoints, human requests | ⬜ |
| 37 | MCP + reference task | ⬜ |

### Part F — Google ADK (kernel: adk)
| # | Topic | |
|---|---|---|
| 38 | Core model — agents, runners, the agent tree | ⬜ |
| 39 | Tools & state — tools, session state | ⬜ |
| 40 | Multi-agent — sub-agents, delegation, agent-as-tool | ⬜ |
| 41 | Workflow control — Sequential / Parallel / Loop agents, callbacks | ⬜ |
| 42 | Persistence & HITL — session services, confirmation | ⬜ |
| 43 | MCP + reference task | ⬜ |

### Part G — Durable execution (kernel: core)
| # | Topic | |
|---|---|---|
| 44 | Durable execution — event history, deterministic replay, workflows vs activities | ⬜ |
| 45 | Temporal hands-on — an agent that survives a crash; signals for human approval | ⬜ |

### Part H — Across frameworks (kernel: core)
| # | Topic | |
|---|---|---|
| 46 | A2A — agent-to-agent protocol (MCP connects agents to tools; A2A connects agents to agents) | ⬜ |
| 47 | The reference task, side by side — comparison matrix and how to choose | ⬜ |
| 48 | Orchestrator-workers with dynamic subtasks | ⬜ |
| 49 | Evaluator-optimizer loops | ⬜ |
| 50 | Cost and latency across a graph — token budget per node | ⬜ |

**Capstone** (kernel: langgraph): the Phase 4 MCP tool mesh orchestrated in
LangGraph, with parallel workers, retries, SQLite checkpoints, a human approval
gate and a live cost meter. The run is killed partway through and resumes, and
one agent is exposed over A2A so a CrewAI or ADK agent can call it.

Notebooks 46–47 run code from several frameworks by launching each in its own
environment as a subprocess, which also shows that A2A works across process and
framework boundaries.

## Layout

```
PHASE_5_ORCHESTRATION/
  orch_lab/            # Phase 5 core (reuses study_buddy, rag_lab, agents_lab, mcp_lab)
    kernels.py         #   which environment am I in? requires(...)
    fakes.py           #   ScriptedLLM, ProcessCrash — deterministic demos
  notebooks/           # 01..50 + capstone
  notes/               # long-form notes on the big ideas
  requirements/        # base.txt + one file per environment
  setup.sh             # builds .venvs/<env> and registers the kernels
  docker-compose.yml   # services only (Temporal)
  .venvs/              # (git-ignored) the environments
```

## Versions this phase was built against

LangGraph 1.2 · OpenAI Agents SDK 0.22 · CrewAI 1.15 · Microsoft Agent
Framework 1.19 · Google ADK 2.9 · Temporal Python SDK 1.33 · Python 3.12.
Frameworks move fast; `requirements/*.txt` set minimum versions, so if an API
has changed, compare the version installed on your machine against this list.

## Deferred to later phases

Long-term memory → Phase 6 · evaluation → Phase 7 · production observability
and security hardening → Phase 8 · architecture → Phase 9 · real systems →
Phase 10.
