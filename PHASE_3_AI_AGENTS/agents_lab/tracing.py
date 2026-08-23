"""Agent tracing: capture the think -> act -> observe trajectory (notebook 18).

Every agent in ``agents_lab`` records its run as a list of :class:`Step` objects
through a shared :class:`Tracer`. This makes an otherwise invisible loop
inspectable: you can print the trajectory, count tool calls, and debug why an
agent went sideways. Deeper production observability (spans, dashboards) is
Phase 8 — this is the minimal, dependency-free version.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator

# The four things an agent does each turn. "action" = a tool call it requested,
# "observation" = what that tool returned.
StepKind = str  # "thought" | "action" | "observation" | "answer" | "error"


@dataclass
class Step:
    kind: StepKind
    content: str
    meta: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        label = self.kind.upper()
        return f"{label:<11} {self.content}"


class Tracer:
    """Collects the ordered steps of a single agent run."""

    def __init__(self) -> None:
        self.steps: list[Step] = []

    def add(self, kind: StepKind, content: str, **meta: Any) -> Step:
        step = Step(kind, str(content), meta)
        self.steps.append(step)
        return step

    # Convenience shortcuts the agent modules call.
    def thought(self, content: str, **meta: Any) -> Step:
        return self.add("thought", content, **meta)

    def action(self, tool: str, arguments: dict[str, Any]) -> Step:
        return self.add("action", f"{tool}({arguments})", tool=tool, arguments=arguments)

    def observation(self, content: str, **meta: Any) -> Step:
        return self.add("observation", content, **meta)

    def answer(self, content: str, **meta: Any) -> Step:
        return self.add("answer", content, **meta)

    def error(self, content: str, **meta: Any) -> Step:
        return self.add("error", content, **meta)

    def pretty(self) -> str:
        """Human-readable trajectory, one step per line."""
        return "\n".join(str(s) for s in self.steps)

    def tool_calls(self) -> list[Step]:
        return [s for s in self.steps if s.kind == "action"]

    def __iter__(self) -> Iterator[Step]:
        return iter(self.steps)

    def __len__(self) -> int:
        return len(self.steps)
