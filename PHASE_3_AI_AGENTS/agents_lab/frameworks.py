"""Framework bridge: from-scratch -> OpenAI Agents SDK / LangGraph (notebook 19).

Everything so far was built by hand so the mechanics are visible. Production code
usually leans on a framework. This module maps our :class:`~agents_lab.tools.Tool`
objects onto two popular ones so you see they're the *same ideas* with different
APIs. Deep framework coverage (LangGraph, Temporal, CrewAI, AutoGen, Semantic
Kernel, ADK) is Phase 5. Imports are lazy and optional: if a framework isn't
installed, the helper raises a clear message instead of failing at import time.
"""

from __future__ import annotations

from typing import Any

from agents_lab.tools import ToolRegistry


def to_openai_agents(
    tools: ToolRegistry,
    instructions: str = "You are a helpful agent.",
    name: str = "agents_lab",
    model: str = "gpt-4o-mini",
) -> Any:
    """Build an OpenAI Agents SDK ``Agent`` from our tool registry.

    Requires ``pip install openai-agents``.
    """
    try:
        from agents import Agent as OAAgent, function_tool
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("Install the OpenAI Agents SDK: pip install openai-agents") from exc

    wrapped = []
    for tname in tools.names:
        tool = tools[tname]
        wrapped.append(function_tool(name_override=tool.name)(tool.func))
    return OAAgent(name=name, instructions=instructions, tools=wrapped, model=model)


def to_langgraph(
    tools: ToolRegistry,
    provider_model: str = "openai:gpt-4o-mini",
) -> Any:
    """Build a minimal prebuilt ReAct graph in LangGraph from our tools.

    Requires ``pip install langgraph langchain``.
    """
    try:
        from langchain_core.tools import StructuredTool
        from langgraph.prebuilt import create_react_agent
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("Install LangGraph: pip install langgraph langchain") from exc

    lc_tools = [
        StructuredTool.from_function(func=tools[n].func, name=tools[n].name, description=tools[n].description)
        for n in tools.names
    ]
    return create_react_agent(provider_model, lc_tools)
