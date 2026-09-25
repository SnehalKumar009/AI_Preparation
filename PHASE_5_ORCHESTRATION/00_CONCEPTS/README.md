# 00 · Concepts: Orchestration Without a Framework

Three notebooks, plain Python, no orchestration framework. They teach the ideas
that every framework section of Phase 5 is built on, so that when you open
LangGraph, CrewAI or any other section, you recognise the machinery.

This section is **optional but recommended**. Every framework section explains
the concepts it uses in its own words and runs on its own, and it links back
here for the deeper version.

| # | Notebook | What you get | Needs |
|---|---|---|---|
| 01 | `01_why_orchestration.ipynb` | A 40-line agent loop, broken five ways: order, parallelism, durability, waiting for humans, replay | nothing (Ollama optional) |
| 02 | `02_building_blocks.ipynb` | A real ~120-line engine: state, reducers, nodes, edges, routers, parallel branches, cycles, SQLite checkpoints, pause/resume, time travel | nothing |
| 03 | `03_workflows_vs_agents.ipynb` | The six patterns from fixed workflow to autonomous agent, built on a real model, plus a checklist for choosing | Ollama |

About 3 hours in total. Notebook 02 is the core; take your time with it.

## Setup

```bash
cd PHASE_5_ORCHESTRATION/00_CONCEPTS
./setup.sh                       # creates .venv and the "Phase 5 · Concepts" kernel
ollama pull qwen3-coder:30b      # for notebook 03 (and the optional cell in 01)
```

Then open the notebooks in Jupyter (`.venv/bin/jupyter lab notebooks`) or VS
Code, and choose the kernel **Phase 5 · Concepts**.

To use a different Ollama model, set `OLLAMA_MODEL` before starting Jupyter,
e.g. `OLLAMA_MODEL=llama3.1:8b`. It must support tool calling and structured
output.

## How to study each notebook

1. Read the **Why this exists** section before running anything.
2. For each demo, **predict the output**, then run it and compare.
3. Read the **trade-offs** and **common mistakes** sections.
4. Answer **Check yourself** before opening the answers. If you miss two or
   more, go back to the matching demo.

## What's inside

```
00_CONCEPTS/
  README.md
  requirements.txt     # ollama, jupyter (that's all)
  setup.sh             # builds .venv, registers the kernel
  notebooks/
    01_why_orchestration.ipynb
    02_building_blocks.ipynb
    03_workflows_vs_agents.ipynb
```

Notebooks write small SQLite/JSON files to `notebooks/checkpoints/` while they
run. The folder is git-ignored and safe to delete.
