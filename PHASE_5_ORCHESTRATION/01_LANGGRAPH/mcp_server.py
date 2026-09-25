"""A small MCP server for this section: the Nimbus Robotics knowledge base.

It exposes the markdown files in ``data/kb/`` as **tools** an agent can call,
plus a place to save finished reports. Notebook 11 connects LangGraph to it,
and the project notebook uses it as the agent's only source of facts.

Run it by itself to check it starts (it then waits for a client on stdin):

    .venv/bin/python mcp_server.py

MCP clients (like ``langchain-mcp-adapters``) launch this file as a subprocess
and talk to it over stdin/stdout, so you normally never run it by hand.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

from mcp.server.fastmcp import FastMCP

HERE = Path(__file__).resolve().parent
KB_DIR = HERE / "data" / "kb"
REPORTS_DIR = HERE / "data" / "reports"

mcp = FastMCP("nimbus-kb", log_level="WARNING")   # keep per-request logs quiet


def _docs() -> dict[str, str]:
    return {p.stem: p.read_text(encoding="utf-8") for p in sorted(KB_DIR.glob("*.md"))}


@mcp.tool()
def list_documents() -> list[dict]:
    """List every document in the Nimbus Robotics knowledge base with its title."""
    return [{"name": name, "title": text.splitlines()[0].lstrip("# ")}
            for name, text in _docs().items()]


@mcp.tool()
def read_document(name: str) -> str:
    """Return the full text of one knowledge-base document, by the name from list_documents."""
    docs = _docs()
    if name not in docs:
        return f"No document named {name!r}. Available: {', '.join(docs)}"
    return docs[name]


@mcp.tool()
def search_documents(query: str, limit: int = 3) -> list[dict]:
    """Keyword search over the knowledge base. Returns the best-matching documents with a snippet."""
    words = [w for w in re.findall(r"\w+", query.lower()) if len(w) > 2]
    scored = []
    for name, text in _docs().items():
        lower = text.lower()
        score = sum(lower.count(w) for w in words)
        if score:
            first = min((lower.find(w) for w in words if w in lower), default=0)
            snippet = text[max(0, first - 80): first + 220].replace("\n", " ")
            scored.append({"name": name, "score": score, "snippet": snippet})
    return sorted(scored, key=lambda d: d["score"], reverse=True)[:limit]


@mcp.tool()
def save_report(title: str, markdown: str) -> str:
    """Save a finished report as a markdown file. Returns the saved file's path."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:60] or "report"
    path = REPORTS_DIR / f"{datetime.now():%Y%m%d-%H%M%S}-{slug}.md"
    path.write_text(f"# {title}\n\n{markdown}\n", encoding="utf-8")
    return str(path.relative_to(HERE))


@mcp.resource("kb://index")
def kb_index() -> str:
    """A plain-text index of the knowledge base (an MCP *resource*: data, not an action)."""
    return "\n".join(f"{d['name']}: {d['title']}" for d in list_documents())


if __name__ == "__main__":
    mcp.run()          # stdio transport by default
