"""Agentic RAG: retrieval as a tool the agent decides to call (notebook 15).

Phase 2 built retrievers; here retrieval becomes just another tool an agent can
invoke when it decides it needs facts. ``make_rag_tool`` wraps any ``rag_lab``
retriever (anything with ``search(query, k)``) into a :class:`~agents_lab.tools.Tool`
that returns ``[source]``-tagged context. ``build_rag_agent`` returns a ready
:class:`~agents_lab.loop.Agent` that grounds its answers in that corpus.
"""

from __future__ import annotations

from typing import Any

from rag_lab.pipeline import format_context

from agents_lab.loop import Agent
from agents_lab.tools import Tool, ToolRegistry

_RAG_SYSTEM = (
    "You are a research agent. Use the `search` tool to look up facts in the "
    "knowledge base, then answer using ONLY what you retrieved and cite [source] "
    "tags. If the search doesn't contain a needed fact, say you don't know."
)


def make_rag_tool(retriever: Any, k: int = 4, name: str = "search") -> Tool:
    """Wrap a rag_lab retriever's ``search`` into an agent tool."""

    def search(query: str) -> str:
        hits = retriever.search(query, k=k)
        return format_context(hits) if hits else "(no results)"

    return Tool(
        name=name,
        description="Search the knowledge base for facts relevant to a query.",
        func=search,
        parameters={
            "type": "object",
            "properties": {"query": {"type": "string", "description": "What to look up"}},
            "required": ["query"],
        },
    )


def build_rag_agent(
    retriever: Any,
    provider: str = "ollama",
    model: str | None = None,
    k: int = 4,
    max_steps: int = 6,
    tracker: Any | None = None,
) -> Agent:
    """An Agent whose only tool is retrieval over ``retriever``."""
    tools = ToolRegistry([make_rag_tool(retriever, k=k)])
    return Agent(
        provider=provider,
        model=model,
        tools=tools,
        system=_RAG_SYSTEM,
        max_steps=max_steps,
        tracker=tracker,
    )
