# 03 · CrewAI

A complete, self-contained course on **CrewAI**: role-playing agents working
as **crews**, and event-driven **Flows** that give you explicit, stateful control
around them. Everything runs on a **local Ollama model**; OpenAI is optional.

Everything this section needs is in this folder: its own environment, model
helper, MCP server and data. Nothing is imported from other phases or other
sections.

## Setup

```bash
cd PHASE_5_ORCHESTRATION/03_CREWAI
./setup.sh                        # creates .venv + the "Phase 5 · CrewAI" Jupyter kernel
ollama pull qwen3-coder:30b       # the default model (needs tool calling)
ollama pull nomic-embed-text      # embeddings for memory (notebook 09)
cp .env.example .env              # optional: a different model, or an OpenAI key
```

Open the notebooks in Jupyter (`.venv/bin/jupyter lab notebooks`) or VS Code
and choose the kernel **Phase 5 · CrewAI**.

Three things every notebook does, and why:

- **`await crew.kickoff_async()`**: inside Jupyter an event loop is already
  running, and this version's synchronous `crew.kickoff()` refuses to run there.
- **`import common` before `crewai`**: `common.py` switches off CrewAI's anonymous
  usage telemetry, which is read when CrewAI is first imported.
- **`quiet_console()`**: hides CrewAI's decorative console panels so the teaching
  output stays readable (notebook 10 turns them back on).

## Curriculum

| # | Notebook | You'll be able to… |
|---|---|---|
| 01 | `01_meet_crewai` | explain Crews vs Flows; run a crew on Ollama |
| 02 | `02_agents_and_tasks` | see the **real prompt** CrewAI builds; design personas, tasks and context |
| 03 | `03_llms_and_structured_output` | connect models; get typed task results |
| 04 | `04_tools` | function vs class tools; recorded tool failures; `result_as_answer` |
| 05 | `05_processes_and_collaboration` | sequential, parallel and hierarchical crews, with measured costs |
| 06 | `06_flows` | explicit control flow: state, routers, joins, guarded loops, crews inside steps |
| 07 | `07_guardrails` | validate task outputs and retry with feedback, in code or with an LLM judge |
| 08 | `08_human_in_the_loop` | pause a flow for a human, save it, resume it from another process |
| 09 | `09_memory_and_persistence` | memory across runs; what survives a real crash |
| 10 | `10_observability` | verbose runs, callbacks, event listeners, log files |
| 11 | `11_mcp_tools` | MCP servers as tools with `MCPServerAdapter`, and a serious pitfall |
| 12 | `12_production_concerns` | how limits really fail; batches; testing with a fake LLM |
| ⭐ | `project` | the Nimbus Briefing Assistant as a Flow with crews: planned, parallel, self-reviewing, crash-tolerant, human-approved |

Plan for about **10–12 hours** in total. Notebooks 06, 08, 09 and 11 are the core.

## Findings you'll see demonstrated

Each of these was observed while building the section, and is reproduced (or
measured) in the notebook. Versions: CrewAI 1.15.

- **The prompt is just your persona.** "You are {role}. {backstory} Your personal goal is: {goal}" (02).
- **`token_usage` accumulates on the LLM object**: one-call runs sharing an LLM reported 1, 2, 3 calls (02).
- **A false alarm from that trap**: a hierarchical crew first measured at "36 calls" was really 3 calls and ~20× the tokens (05).
- **CrewAI puts the JSON schema, descriptions included, into the prompt**, so field descriptions *are* read here, unlike the Ollama structured-output path in the other sections (03).
- **Tool failures are recorded** (`has_tool_failures`, `tool_failures`); the tool cache didn't prevent repeated calls (04).
- **A persona can defeat a guardrail**: a "wordy marketer" failed a 20-word rule on every retry, then the task raised a plain `Exception` (07).
- **`@human_feedback` reads as `last_human_feedback`** (the docstring example is wrong), and an LLM maps "maybe later" to *rejected* (08).
- **Deep memory recall missed a stored fact** that shallow vector search found, after the LLM misfiled it (09).
- **`@persist` survived a real kill but restarts from `@start`**, so steps must be idempotent. **Crew checkpoints didn't survive our crash tests**, and failed silently with function tools (09).
- **`step_callback` missed tool calls; `output_log_file` silently appends `.txt`** (10).
- **`Agent(mcps=...)` in a crew exposed every server tool under hashed names, invisible to `agent.tools`, and wrote reports on its own** (11).
- **Leaked tool calls** (a tool call written as text) became a "customer note" until a guardrail stopped them (11).
- **Hitting `max_iter` returns a guess, not an error** ("10" after 3 of 10 steps); `max_execution_time` fired at ~8 s for a 3 s limit (12).

## How to study each notebook

Every notebook has the same shape: **why** → **how it works** (with a diagram) →
**annotated code** (predict each output before running it) → **trade-offs and
common mistakes** → **check yourself** (answer before opening the hidden
answers). Model output varies, so your wording will differ, but every behaviour
described was observed on the default model.

For the concepts without any framework, see `../00_CONCEPTS/`. The project
notebook ends with a side-by-side comparison with the LangGraph and OpenAI Agents
SDK versions.

## What's in this folder

```
03_CREWAI/
  README.md
  requirements.txt     # crewai, crewai-tools[mcp], mcp, ollama (for embeddings), Jupyter
  setup.sh             # builds .venv and registers the kernel
  .env.example         # model provider, keys, embedding model (copy to .env)
  common.py            # get_llm(), embedder_config(), quiet_console(), reject_leaked_tool_calls()
  mcp_server.py        # the Nimbus knowledge-base MCP server (notebook 11 + project)
  data/kb/             # six documents about the fictional company Nimbus Robotics
  notebooks/           # 01–12 + project
```

Notebooks write scratch files to `notebooks/checkpoints/`, and reports go to
`data/reports/`. Both are git-ignored and safe to delete.

## Versions this section was built and tested with

crewai 1.15 · crewai-tools 1.15 · mcp 1.28 · Python 3.12 · Ollama with
`qwen3-coder:30b` and `nomic-embed-text`. CrewAI changes quickly; several findings
above are version-specific, so re-test them if your version differs.
