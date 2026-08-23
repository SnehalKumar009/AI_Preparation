"""The agent loop: think -> act -> observe, repeat (notebooks 01-03).

This is the beating heart of an agent. Given a task, the model may call tools;
each tool result is fed back as an observation and the loop continues until the
model answers or a step cap is hit. It uses the provider's **native tool
calling** (``chat(tools=...)``) to get structured calls, then feeds results back
as plain text so the loop stays portable across Ollama / OpenAI / Anthropic.

Cost is tracked on every call via ``study_buddy.CostTracker``; the whole
trajectory is captured in a :class:`~agents_lab.tracing.Tracer`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from study_buddy import CostTracker, get_provider

from agents_lab.tools import ToolRegistry, default_tools
from agents_lab.tracing import Step, Tracer

DEFAULT_SYSTEM = (
    "You are a helpful agent. You may call the provided tools to gather facts or "
    "do computations. Think step by step. When you have enough information, reply "
    "with a final answer in plain text and stop calling tools."
)


@dataclass
class AgentResult:
    answer: str
    steps: list[Step] = field(default_factory=list)
    tracker: CostTracker | None = None

    @property
    def n_tool_calls(self) -> int:
        return sum(1 for s in self.steps if s.kind == "action")


class Agent:
    """A tool-using agent driven by native function calling."""

    def __init__(
        self,
        provider: str = "ollama",
        model: str | None = None,
        tools: ToolRegistry | None = None,
        system: str = DEFAULT_SYSTEM,
        max_steps: int = 6,
        temperature: float = 0.0,
        tracker: CostTracker | None = None,
    ) -> None:
        self.llm = get_provider(provider)
        self.model = model
        self.tools = tools if tools is not None else default_tools()
        self.system = system
        self.max_steps = max_steps
        self.temperature = temperature
        self.tracker = tracker if tracker is not None else CostTracker()

    def run(self, task: str, tracer: Tracer | None = None) -> AgentResult:
        tracer = tracer or Tracer()
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system},
            {"role": "user", "content": task},
        ]

        for _ in range(self.max_steps):
            resp = self.llm.chat(
                messages,
                model=self.model,
                temperature=self.temperature,
                tools=self.tools.schemas,
            )
            self.tracker.add(resp)

            if not resp.tool_calls:
                tracer.answer(resp.text)
                return AgentResult(resp.text, tracer.steps, self.tracker)

            if resp.text:
                tracer.thought(resp.text)
            # Execute every requested call and feed results back as one message.
            observations = []
            for call in resp.tool_calls:
                name, args = call["name"], call.get("arguments", {})
                tracer.action(name, args)
                result = self.tools.dispatch(name, args)
                tracer.observation(result, tool=name)
                observations.append(f"Tool `{name}` returned: {result}")

            messages.append({"role": "assistant", "content": resp.text or "(calling tools)"})
            messages.append(
                {"role": "user", "content": "\n".join(observations) + "\n\nContinue or give the final answer."}
            )

        # Step cap reached: force a final answer with no more tools.
        final = self.llm.chat(
            messages + [{"role": "user", "content": "Stop. Give your best final answer now."}],
            model=self.model,
            temperature=self.temperature,
        )
        self.tracker.add(final)
        tracer.answer(final.text)
        return AgentResult(final.text, tracer.steps, self.tracker)
