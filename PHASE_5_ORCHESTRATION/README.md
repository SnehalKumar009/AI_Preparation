# Phase 5 · Orchestration & Frameworks

A Phase 3-style agent is a loop: the model decides every next step, and the
work lives in one process's memory. That's why it can't guarantee order, run
steps in parallel, survive a crash, wait hours for a human, or replay a step.
**Orchestration** fixes all five by making the structure explicit: **state,
nodes, edges and checkpoints**.

Phase 5 teaches orchestration through the major frameworks, **one self-contained
section per framework**.

## Sections

| Section | What it is | Status |
|---|---|---|
| `00_CONCEPTS/` | The ideas in plain Python: why orchestration, a hand-built engine, workflows vs agents | ✅ ready |
| `01_LANGGRAPH/` | LangGraph: 12 notebooks + project | ✅ ready |
| `02_OPENAI_AGENTS_SDK/` | OpenAI Agents SDK: 12 notebooks + project | ✅ ready |
| `03_CREWAI/` | CrewAI: 12 notebooks + project | ✅ ready |
| `04_MICROSOFT_AGENT_FRAMEWORK/` | Microsoft Agent Framework (successor to AutoGen and Semantic Kernel) | planned |
| `05_GOOGLE_ADK/` | Google Agent Development Kit | planned |
| `06_TEMPORAL/` | Durable execution with Temporal | planned |
| `07_COMPARISON/` | The same project in every framework, side by side, plus how to choose; A2A | planned |

## Rules every section follows

- **Self-contained.** Each folder has its own `requirements.txt`, `setup.sh`,
  `.venv`, Jupyter kernel, `.env`, MCP server and data. Nothing is imported from
  Phases 1–4 or from another section. Frameworks pin conflicting versions of
  shared libraries, so separate environments are required, not just tidy.
- **No book needed.** Every notebook explains why, then how (with a diagram),
  then annotated code, trade-offs and mistakes, and check-yourself questions.
- **Local first.** Everything runs on Ollama (`qwen3-coder:30b` by default);
  cloud providers are optional.
- **Same arc, same project.** Framework sections follow the same order of
  topics and end with the same project, the **Nimbus Briefing Assistant**, so
  frameworks can be compared on identical work.

## Where to start

- New to orchestration → `00_CONCEPTS/` (about 3 hours), then any framework.
- Want to get building → `01_LANGGRAPH/`. Its notebooks explain each concept
  where it's needed.

Each section's README has its setup steps and curriculum.
