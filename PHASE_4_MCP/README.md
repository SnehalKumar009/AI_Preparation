# Local MCP — Phase 4

Go from **MCP-from-scratch → a local tool mesh an agent can drive**. The **Model
Context Protocol (MCP)** is a standard way for an LLM host to discover and call
tools exposed by independent **servers**. Phase 3 taught you to hand-write tools
and wire them into an agent loop; this phase re-expresses that idea as the
protocol the ecosystem is standardizing on — one concept per notebook — then
puts an agent on top of five real local servers.

Everything is built **from scratch first** so the wire format stays visible
(JSON-RPC 2.0 over stdio, a minimal server and client), and only then mapped onto
the **official `mcp` SDK**. It **reuses Phase 1 `study_buddy`** (providers + cost),
**Phase 2 `rag_lab`** (retrieval + guardrails for the code-search server), and
**Phase 3 `agents_lab`** (the agent loop the MCP tools are bridged into).

Every notebook names which client it uses and can run against **DeepSeek /
OpenAI / Anthropic / oxalpha / Ollama interchangeably** — swap one string in
`get_provider(...)`; missing keys fall back to Ollama.

## Setup (.venv, no Docker)

```bash
# from the AI_Preparation root, reuse the existing venv (or create one)
python3 -m venv .venv && source .venv/bin/activate

# install the Phase 4 superset (also covers Phases 1–3 deps)
pip install -r PHASE_4_MCP/requirements.txt

# local models (optional — every notebook can fall back to Ollama)
ollama pull qwen3-coder:30b
ollama pull nomic-embed-text        # code-search server (notebook 22)
```

Cloud keys live in `PHASE_1_LLM_FUNDAMENTALS/.env` (reused). Missing keys skip cleanly.

## How to run

```bash
jupyter lab PHASE_4_MCP/notebooks
```

Work through `01_*.ipynb` → `37_*.ipynb`, then `capstone.ipynb`.

## Curriculum (one concept per notebook)

### Foundations — protocol mechanics (from scratch)
| # | Topic | # | Topic |
|---|-------|---|-------|
| 01 | What is MCP (host/client/server) | 05 | The `initialize` handshake |
| 02 | Primitives: tools/resources/prompts | 06 | Minimal server from scratch |
| 03 | Transports: stdio vs HTTP/SSE | 07 | Minimal client from scratch |
| 04 | JSON-RPC 2.0 basics | — | — |

### Official SDK & primitives in depth
| # | Topic | # | Topic |
|---|-------|---|-------|
| 08 | FastMCP server (official SDK) | 11 | Resources in depth |
| 09 | MCP client with the SDK | 12 | Prompts in depth |
| 10 | Tools in depth (schemas) | — | — |

### The five servers (Project 6)
| # | Topic | # | Topic |
|---|-------|---|-------|
| 13 | Filesystem server | 16 | Terminal server |
| 14 | Git server | 17 | Calculator server |
| 15 | SQLite server | — | — |

### Integration
| # | Topic | # | Topic |
|---|-------|---|-------|
| 18 | Multi-server client | 21 | Demo: "Find every TODO" |
| 19 | Bridging MCP into the agent loop | 22 | Code-search server (Code RAG) |
| 20 | Provider interchangeability | — | — |

### Security & deployment
| # | Topic | # | Topic |
|---|-------|---|-------|
| 23 | Prompt injection via tool output | 26 | Secret / PII redaction |
| 24 | Filesystem sandboxing | 27 | Deployment & config (`mcp.json`) |
| 25 | Terminal allowlisting | — | — |

### Advanced MCP (full spec surface)
| # | Topic | # | Topic |
|---|-------|---|-------|
| 28 | Sampling (server → host LLM) | 33 | Pagination & completion |
| 29 | Roots (client filesystem scope) | 34 | Tool annotations & safety hints |
| 30 | Progress & cancellation | 35 | Streamable HTTP & remote servers |
| 31 | Server logging | 36 | Auth for remote servers (OAuth 2.1) |
| 32 | List-changed & subscriptions | 37 | Debugging with MCP Inspector |

**Capstone:** a provider-interchangeable agent over all five servers with a live
USD cost meter — ask *"Find every TODO in the repository"* and watch it choose
tools across servers.

## Layout

```
PHASE_4_MCP/
  mcp_lab/            # Phase 4 core (reuses study_buddy + rag_lab + agents_lab)
    protocol.py      # JSON-RPC 2.0 message helpers
    transport.py     # stdio (newline-delimited JSON) framing
    miniserver.py    # minimal MCP server, from scratch
    miniclient.py    # minimal MCP client, from scratch
    advanced.py      # bidirectional session: sampling/roots/progress/logging
    client.py        # multi-server client (namespaced tools)
    agent_bridge.py  # MCP tools -> agents_lab ToolRegistry / Agent
    safety.py        # path/command guards + reused rag_lab guardrails
    servers/         # filesystem, git, sqlite, terminal, calculator, code_search
  notebooks/         # 01..37 + capstone
  requirements.txt   # superset of Phases 1–3
```

## Provider mix & tool-calling matrix

Not every provider supports native tool-calling equally. The agent bridge relies
on the normalized `ChatResponse.tool_calls`, so swapping providers is a one-string
change — but pick a tool-capable model:

| Provider | Tool calling | Notes |
|----------|--------------|-------|
| Ollama (`qwen3-coder`) | ✅ | Default; fully local |
| OpenAI | ✅ | `gpt-4o-mini` etc. |
| DeepSeek | ✅ | OpenAI-compatible endpoint |
| Anthropic | ✅ | Different wire format, handled in the provider |
| oxalpha (OpenRouter) | ⚠️ | Depends on the underlying routed model |

## Running a server by hand

```bash
python -m mcp_lab.servers.calculator     # speaks JSON-RPC over stdio
```

Servers are read-only and sandboxed by default (see notebooks 23–27):

- **filesystem** — confined to `MCP_FS_ROOT`; `../` traversal blocked.
- **terminal** — allowlisted programs only, no shell, with a timeout.
- **sqlite** — `mode=ro` connection; `SELECT`-only.
- **git** — status/log/diff/blame only; never writes.

## Reuse from earlier phases

- **Phase 1 `study_buddy`** — `get_provider`, `CostTracker` (capstone cost meter).
- **Phase 2 `rag_lab`** — `chunking` + `NumpyStore` (code-search), `guardrails`
  (`redact_pii`, `detect_injection`) via `mcp_lab.safety`.
- **Phase 3 `agents_lab`** — `Agent`, `ToolRegistry`, `Tool` (the agent bridge).

## Deferred to later phases

Deep orchestration & LangGraph → Phase 5 · persistent memory → Phase 6 ·
evaluation → Phase 7 · production/observability/security hardening → Phase 8 ·
architecture → Phase 9 · full coding assistant → Phase 10.
