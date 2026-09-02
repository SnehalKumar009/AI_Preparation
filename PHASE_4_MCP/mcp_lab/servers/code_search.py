"""Code-search MCP server (notebook 22) — the Project 3 "Code RAG" fold-in.

Indexes a code repository and answers "where is X?" questions. It reuses Phase 2
``rag_lab`` for chunking + semantic retrieval, so code RAG is almost free once the
doc-RAG machinery exists. Two tools cover both retrieval styles:

- ``search_code`` — semantic search over embedded code chunks (needs Ollama's
  ``nomic-embed-text``); falls back to a substring scan if embeddings are
  unavailable, so the server is always useful.
- ``grep_code`` — exact substring/regex-free scan with file:line hits.

The code root comes from ``MCP_CODE_ROOT`` (default: this phase's folder). The
semantic index is built lazily on first ``search_code`` call and cached.
"""

from __future__ import annotations

import os
from pathlib import Path

from mcp_lab.miniserver import MiniMCPServer

_ROOT = Path(os.getenv("MCP_CODE_ROOT", Path(__file__).resolve().parent.parent.parent)).resolve()
_CODE_GLOBS = ("*.py", "*.md", "*.txt", "*.js", "*.ts", "*.yaml", "*.yml", "*.toml")
_MAX_FILES = 300

_store = None  # cached rag_lab NumpyStore (lazy)
_indexed = False


def _iter_code_files() -> list[Path]:
    files: list[Path] = []
    for pattern in _CODE_GLOBS:
        files.extend(_ROOT.rglob(pattern))
    files = [f for f in files if f.is_file() and ".git" not in f.parts][:_MAX_FILES]
    return files


def _build_index():
    """Build a semantic index over code chunks; return None if embeddings fail."""
    global _store, _indexed
    if _indexed:
        return _store
    _indexed = True
    try:
        from rag_lab.chunking import fixed_chunks
        from rag_lab.vectorstore import NumpyStore

        store = NumpyStore(provider="ollama")
        chunks = []
        for path in _iter_code_files():
            text = path.read_text(encoding="utf-8", errors="replace")
            rel = str(path.relative_to(_ROOT))
            chunks.extend(fixed_chunks(text, source=rel, chunk_size=120, overlap=20))
        if chunks:
            store.add(chunks)
        _store = store
    except Exception:
        _store = None  # embeddings/Ollama unavailable -> grep fallback
    return _store


def search_code(query: str, k: int = 5) -> str:
    """Semantically search code for a concept; falls back to grep if needed."""
    store = _build_index()
    if store is None or len(store) == 0:
        return grep_code(query)
    hits = store.search(query, k=int(k))
    if not hits:
        return "(no matches)"
    return "\n\n".join(f"[{h.chunk.source}] (score {h.score:.2f})\n{h.chunk.text}" for h in hits)


def grep_code(text: str) -> str:
    """Exact substring scan returning file:line matches."""
    needle = text.lower()
    out: list[str] = []
    for path in _iter_code_files():
        rel = str(path.relative_to(_ROOT))
        try:
            for i, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                if needle in line.lower():
                    out.append(f"{rel}:{i}: {line.strip()}")
                    if len(out) >= 100:
                        return "\n".join(out)
        except OSError:
            continue
    return "\n".join(out) if out else "(no matches)"


def build_server() -> MiniMCPServer:
    server = MiniMCPServer("code_search")
    server.tool(
        name="search_code",
        description="Semantically search the codebase for a concept (e.g. 'where is auth').",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Natural-language query"},
                "k": {"type": "integer", "description": "How many hits (default 5)"},
            },
            "required": ["query"],
        },
    )(search_code)
    server.tool(
        name="grep_code",
        description="Exact substring scan of the codebase, returning file:line hits.",
        input_schema={
            "type": "object",
            "properties": {"text": {"type": "string", "description": "Substring to find"}},
            "required": ["text"],
        },
    )(grep_code)
    return server


if __name__ == "__main__":
    build_server().serve_stdio()
