# LLM Fundamentals — Study Buddy (Phase 1)

A provider-agnostic LLM toolkit for learning the 15 Phase 1 fundamentals.
Swap between local **Ollama** and cloud (**OpenAI / Gemini / DeepSeek**) with one
line, track **real USD cost**, and build up to a RAG + tools chat app.

## Setup (Ubuntu)

```bash
# 1. Install deps (use a venv)
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Pull local models
ollama pull qwen3-coder:30b
ollama pull nomic-embed-text

# 3. Configure keys
cp .env.example .env    # then edit .env with your keys + OLLAMA_HOST
```

## Learn (notebooks)

Work through `notebooks/01_*.ipynb` → `15_*.ipynb`, one topic per session,
then `capstone.ipynb`.

| # | Topic | # | Topic |
|---|-------|---|-------|
| 01 | Tokens | 09 | Prompt Engineering |
| 02 | Cost & Pricing | 10 | Function Calling |
| 03 | Context Window | 11 | Structured Output |
| 04 | Embeddings | 12 | Reasoning vs Non |
| 05 | Roles | 13 | Latency/Cost/Quality |
| 06 | Temperature | 14 | Hallucination |
| 07 | Top-p | 15 | Fine-tune vs Prompt vs RAG |
| 08 | Streaming | — | Capstone |

## Run the WebUI

```bash
streamlit run webui/app.py
```

## Run with Docker

Ollama keeps running on the **host** (so your pulled models are reused); the
containers reach it via `host.docker.internal`.

```bash
cp .env.example .env        # add your cloud keys
docker compose up --build   # app -> :8501, jupyter -> :8888
```

- Streamlit UI: http://localhost:8501
- Jupyter notebooks: http://localhost:8888 (no token)

`notes/`, `notebooks/`, and `pricing.yaml` are mounted, so edits persist without
a rebuild. To run only one service: `docker compose up app` or `... up jupyter`.

## Layout

```
study_buddy/        # shared core (imported by notebooks + webui)
  providers/        # ollama, openai, gemini, deepseek behind one interface
  pricing.yaml      # editable USD/1M-token rates
  cost.py rag.py tokens.py tools.py
notebooks/          # 01..15 + capstone
webui/app.py        # Streamlit capstone
notes/              # your RAG corpus (edit/add .md files)
```

Only `qwen3-coder:30b`? Every notebook runs on Ollama alone; cloud cells are
optional and skip cleanly if a key is missing.
