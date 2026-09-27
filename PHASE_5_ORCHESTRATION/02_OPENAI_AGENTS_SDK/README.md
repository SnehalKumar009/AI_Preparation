# 02 · OpenAI Agents SDK

A complete, self-contained course on the **OpenAI Agents SDK** (`openai-agents`):
a small framework of primitives (agents, the Runner, tools, handoffs,
guardrails, sessions, tracing) plus plain Python for everything else. Despite
the name, everything here runs on a **local Ollama model**; OpenAI is optional.

Everything this section needs is in this folder: its own environment, model
helper, MCP server and data. Nothing is imported from other phases or other
sections.

## Setup

```bash
cd PHASE_5_ORCHESTRATION/02_OPENAI_AGENTS_SDK
./setup.sh                        # creates .venv + the "Phase 5 · OpenAI Agents SDK" Jupyter kernel
ollama pull qwen3-coder:30b       # the default local model (needs tool calling)
cp .env.example .env              # optional: a different model, or an OpenAI key
```

Open the notebooks in Jupyter (`.venv/bin/jupyter lab notebooks`) or VS Code
and choose the kernel **Phase 5 · OpenAI Agents SDK**.

In notebooks, always use `await Runner.run(...)`. `Runner.run_sync` doesn't work
inside Jupyter, because a notebook already runs an event loop.

**Tracing:** the SDK uploads traces to OpenAI by default. `common.py` turns
that off unless `OPENAI_API_KEY` is set, and notebook 10 shows how to record
traces locally instead.

## Curriculum

| # | Notebook | You'll be able to… |
|---|---|---|
| 01 | `01_meet_the_sdk` | explain the primitives and the agent loop; run agents on Ollama |
| 02 | `02_agents_and_the_runner` | configure agents; know exactly when a run ends; pass context the model can't see |
| 03 | `03_models_and_structured_output` | connect any OpenAI-compatible model; get typed results that actually follow your rubric |
| 04 | `04_tools` | write good tools; choose report-vs-raise on errors; timeouts; `tool_use_behavior`; admin-only tools; agents as tools |
| 05 | `05_orchestrating_in_code` | chains, routers, parallel agents, map-reduce and refine loops in plain Python |
| 06 | `06_handoffs_and_multi_agent` | triage with handoffs; filter what the next agent sees; **verify** claimed actions |
| 07 | `07_guardrails` | input, output and tool guardrails, and why input guardrails must block when tools have side effects |
| 08 | `08_human_in_the_loop` | approval-gated tools; pause to JSON and resume in another process; conditional approval |
| 09 | `09_sessions_and_state` | conversation memory; exactly what survives a killed process |
| 10 | `10_streaming_tracing_and_hooks` | stream tokens and events; local tracing; hooks for metrics |
| 11 | `11_mcp_tools` | MCP servers as tools; filtering; approvals; handling leaked tool calls |
| 12 | `12_production_concerns` | error handlers, model retries, budgets, testing with a scripted fake model |
| ⭐ | `project` | the Nimbus Briefing Assistant: planned, parallel, self-reviewing, crash-tolerant, human-approved |

Plan for about **10–12 hours** in total. Notebooks 05, 06, 07 and 08 are the core.

## Findings you'll see demonstrated

These notebooks don't just show the happy path. Each of these was observed
while building them, and is reproduced (or measured) in the notebook:

- **Structured output ignores field descriptions on Ollama.** Rubrics go in the instructions (03).
- **Ollama ignores `tool_choice='required'`** (02).
- **A tool exception is reported to the model by default**, the opposite of LangGraph (04).
- **Batched tool calls ignore "first A, then B" in the prompt.** Order belongs in code (04, 05).
- **Receiving agents get confused by handoff traffic**, and **agents claim actions they never took**. Filter and verify (06, 11).
- **A parallel input guardrail tripped *after* the file was deleted** (07).
- **Conditional approval fails safe** when argument types would be coerced (08).
- **Sessions save after every turn**: completed work survives a kill, and the in-flight step re-runs (09).
- **Local models sometimes "leak" tool calls as text** (2 of 8 runs), so detect and retry (11).
- **Model retries are off by default**, and the policy decides what's retryable (12).

## How to study each notebook

Every notebook has the same shape: **why** → **how it works** (with a diagram) →
**annotated code** (predict each output before running it) → **trade-offs and
common mistakes** → **check yourself** (answer before opening the hidden
answers). Model output varies, so your wording will differ, but every behaviour
described was observed on the default model.

For the concepts without any framework, see `../00_CONCEPTS/`. If you've done
`../01_LANGGRAPH/`, the project notebook ends with a side-by-side comparison.

## What's in this folder

```
02_OPENAI_AGENTS_SDK/
  README.md
  requirements.txt     # openai-agents, mcp, Jupyter
  setup.sh             # builds .venv and registers the kernel
  .env.example         # model provider and key (copy to .env)
  common.py            # get_model(), plus run_agent() (retries leaked tool calls)
  mcp_server.py        # the Nimbus knowledge-base MCP server (notebook 11 + project)
  data/kb/             # six documents about the fictional company Nimbus Robotics
  notebooks/           # 01–12 + project
```

Notebooks write scratch files to `notebooks/checkpoints/`, and reports go to
`data/reports/`. Both are git-ignored and safe to delete.

## Versions this section was built and tested with

openai-agents 0.22 · openai 3.19 · mcp 1.30 · Python 3.12 · Ollama with
`qwen3-coder:30b`. The SDK moves quickly; if an API differs on your machine,
compare your installed version against this list.
