"""Tools + a tool registry (notebooks 02-03).

An agent is only as capable as the tools it can call. A :class:`Tool` bundles a
Python function with the OpenAI-style JSON schema the model sees. A
:class:`ToolRegistry` holds many tools and knows how to:

- hand the model all schemas (``registry.schemas``),
- run a single call the model requested (``dispatch``) with error + timeout
  handling so a broken tool becomes an observation instead of a crash,
- run several independent calls at once (``run_parallel``).

The example tools reuse Phase 1's ``study_buddy.tools`` so we don't reinvent
a calculator.
"""

from __future__ import annotations

import concurrent.futures as _cf
import inspect
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class Tool:
    name: str
    description: str
    func: Callable[..., Any]
    parameters: dict[str, Any] = field(
        default_factory=lambda: {"type": "object", "properties": {}}
    )
    timeout_s: float | None = None

    @property
    def schema(self) -> dict[str, Any]:
        """OpenAI-style function schema (also accepted by Ollama/Anthropic)."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    """A named collection of tools with safe dispatch."""

    def __init__(self, tools: list[Tool] | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        for t in tools or []:
            self.add(t)

    def add(self, tool: Tool) -> Tool:
        self._tools[tool.name] = tool
        return tool

    def register(
        self,
        func: Callable[..., Any] | None = None,
        *,
        name: str | None = None,
        description: str | None = None,
        parameters: dict[str, Any] | None = None,
        timeout_s: float | None = None,
    ) -> Callable[..., Any]:
        """Decorator form: ``@registry.register(description=..., parameters=...)``."""

        def wrap(fn: Callable[..., Any]) -> Callable[..., Any]:
            self.add(
                Tool(
                    name=name or fn.__name__,
                    description=description or (inspect.getdoc(fn) or "").split("\n")[0],
                    func=fn,
                    parameters=parameters or {"type": "object", "properties": {}},
                    timeout_s=timeout_s,
                )
            )
            return fn

        return wrap(func) if func else wrap

    @property
    def schemas(self) -> list[dict[str, Any]]:
        return [t.schema for t in self._tools.values()]

    @property
    def names(self) -> list[str]:
        return list(self._tools)

    def dispatch(self, name: str, arguments: dict[str, Any]) -> str:
        """Execute one tool call. Errors and timeouts come back as text so the
        agent can read them as an observation and recover."""
        tool = self._tools.get(name)
        if tool is None:
            return f"Error: unknown tool '{name}'. Available: {self.names}"
        try:
            if tool.timeout_s is None:
                return str(tool.func(**arguments))
            with _cf.ThreadPoolExecutor(max_workers=1) as ex:
                return str(ex.submit(lambda: tool.func(**arguments)).result(tool.timeout_s))
        except _cf.TimeoutError:
            return f"Error: tool '{name}' timed out after {tool.timeout_s}s"
        except Exception as exc:
            return f"Error running {name}: {exc}"

    def run_parallel(self, calls: list[dict[str, Any]]) -> list[str]:
        """Run several ``{'name', 'arguments'}`` calls concurrently, order kept."""
        with _cf.ThreadPoolExecutor(max_workers=min(8, len(calls) or 1)) as ex:
            futures = [ex.submit(self.dispatch, c["name"], c.get("arguments", {})) for c in calls]
            return [f.result() for f in futures]

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __getitem__(self, name: str) -> Tool:
        return self._tools[name]

    def __len__(self) -> int:
        return len(self._tools)


def default_tools() -> ToolRegistry:
    """A ready-made registry (calculator + current_time) reused from Phase 1."""
    from study_buddy.tools import calculator, current_time

    return ToolRegistry(
        [
            Tool(
                name="calculator",
                description="Evaluate a basic arithmetic expression, e.g. '2 * (3 + 4)'.",
                func=calculator,
                parameters={
                    "type": "object",
                    "properties": {"expression": {"type": "string", "description": "Math expression"}},
                    "required": ["expression"],
                },
            ),
            Tool(
                name="current_time",
                description="Get the current local date and time in ISO format.",
                func=current_time,
                parameters={
                    "type": "object",
                    "properties": {"timezone": {"type": "string", "description": "Ignored; local time"}},
                },
            ),
        ]
    )
