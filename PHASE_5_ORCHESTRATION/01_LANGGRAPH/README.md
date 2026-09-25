# 01 · LangGraph

A complete, self-contained course on **LangGraph**, the low-level orchestration
framework from LangChain. It covers state, nodes and edges, tools and agents,
control flow, multi-agent systems, human-in-the-loop, persistence, streaming,
subgraphs, MCP, and production concerns. It ends with a project that uses all of
them.

Everything this section needs is in this folder: its own environment, model
helper, MCP server and data. Nothing is imported from other phases or other
sections.

## Setup

```bash
cd PHASE_5_ORCHESTRATION/01_LANGGRAPH
./setup.sh                        # creates .venv + the "Phase 5 · LangGraph" Jupyter kernel
ollama pull qwen3-coder:30b       # the default local model (needs tool calling)
cp .env.example .env              # optional: cloud keys, a different model, LangSmith tracing
```

Open the notebooks in Jupyter (`.venv/bin/jupyter lab notebooks`) or VS Code
and choose the kernel **Phase 5 · LangGraph**.

Any Ollama model that supports tool calling works: set `OLLAMA_MODEL` in
`.env`. To use a cloud model, set `MODEL_PROVIDER=openai` or `anthropic` plus
the matching key. Every notebook calls `get_model()` from `common.py`, so this
one setting switches the whole section.

## Curriculum

| # | Notebook | You'll be able to… | Needs |
|---|---|---|---|
| 01 | `01_meet_langgraph` | explain what LangGraph is; build and run a graph | Ollama (§4) |
| 02 | `02_state_nodes_edges` | design state and reducers; route with conditional edges; avoid the fan-in trap | offline |
| 03 | `03_chat_models_and_messages` | use any provider through one interface; messages; structured output that actually follows your rubric | Ollama |
| 04 | `04_tools_and_react_agent` | define tools; build the ReAct loop as a graph; handle tool errors; use `create_agent` | Ollama |
| 05 | `05_control_flow` | LLM routers, `Command`, guarded loops, parallel fan-out, map-reduce with `Send` | Ollama |
| 06 | `06_multi_agent` | build supervisor and handoff systems; isolate each agent's context | Ollama |
| 07 | `07_human_in_the_loop` | pause with `interrupt()`; approve/edit/reject tool calls; resume | Ollama |
| 08 | `08_persistence_and_time_travel` | SQLite checkpoints; survive real process crashes; replay and fork | mostly offline |
| 09 | `09_streaming_and_debugging` | stream tokens, progress and state; debug and trace runs | Ollama |
| 10 | `10_subgraphs` | compose graphs; interrupts inside subgraphs | offline |
| 11 | `11_mcp_tools` | connect graphs to MCP servers; async tools; resources; security | Ollama |
| 12 | `12_production_concerns` | retries, timeouts, error handlers, fallbacks, budgets, caching, testing | mostly offline |
| ⭐ | `project` | build the Nimbus Briefing Assistant using everything above | Ollama |

Plan for about **12–15 hours** in total. Notebooks 02, 05, 07 and 08 are the core;
take your time there.

## How to study each notebook

Every notebook has the same shape:

1. **Why this exists**: read it before running anything.
2. **How it works**: the mechanism, with a diagram.
3. **Annotated code**: *predict the output before running each cell*, then compare.
4. **Trade-offs and common mistakes**.
5. **Check yourself**: answer before opening the hidden answers. If you miss
   two or more, go back to the matching section.

Several notebooks include a **"lesson from building this demo"**: a real mistake
we made while writing it, and how we fixed it. Those are some of the most useful
parts. Model output varies, so your runs will differ in wording, but every
behaviour described was observed on the default model.

If you want the concepts without any framework first, see
`../00_CONCEPTS/`. Each LangGraph notebook stands on its own, though.

## What's in this folder

```
01_LANGGRAPH/
  README.md
  requirements.txt     # LangGraph, LangChain chat models, MCP, Jupyter (versions in the file)
  setup.sh             # builds .venv and registers the kernel
  .env.example         # model provider, keys, tracing (copy to .env)
  common.py            # get_model(): one place to choose the model
  mcp_server.py        # the Nimbus knowledge-base MCP server (notebook 11 + project)
  data/kb/             # six documents about the fictional company Nimbus Robotics
  notebooks/           # 01–12 + project
```

While running, notebooks write scratch files to `notebooks/checkpoints/`, and
the project saves reports to `data/reports/`. Both are git-ignored and safe to
delete.

## Versions this section was built and tested with

LangGraph 1.2 · LangChain 1.4 · langchain-mcp-adapters 0.3 · mcp 1.30 ·
Python 3.12 · Ollama with `qwen3-coder:30b`.

`mcp` is pinned below 2.0 because `langchain-mcp-adapters` 0.3 doesn't work
with it yet (an import error). If a newer adapter supports `mcp` 2.x, the pin
can be lifted.
