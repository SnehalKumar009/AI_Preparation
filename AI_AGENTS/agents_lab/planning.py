"""Planning: decompose first, then act (notebooks 06-07).

Three planning styles:

- ``decompose`` — turn a fuzzy task into an ordered list of concrete sub-steps.
- ``PlanAndExecute`` — make a plan, then run each step with a tool-using Agent.
- ``rewoo`` — *Reasoning WithOut Observation*: plan **all** tool calls up front
  in one shot, execute them, then compose the answer from the results. Fewer LLM
  calls than ReAct because the model doesn't re-plan after every observation.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from study_buddy import CostTracker, get_provider

from agents_lab.loop import Agent, AgentResult
from agents_lab.tools import ToolRegistry, default_tools
from agents_lab.tracing import Tracer

_PLAN_SYSTEM = (
    "Break the user's task into a short ordered list of concrete sub-steps. "
    "Reply with ONLY a JSON array of strings, no prose."
)


def _parse_list(text: str) -> list[str]:
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if not m:
        return [line.strip("-* ").strip() for line in text.splitlines() if line.strip()]
    try:
        return [str(x) for x in json.loads(m.group())]
    except json.JSONDecodeError:
        return []


def decompose(
    task: str,
    provider: str = "ollama",
    model: str | None = None,
    tracker: Any | None = None,
) -> list[str]:
    """Return an ordered list of sub-steps for a task."""
    resp = get_provider(provider).chat(
        [{"role": "system", "content": _PLAN_SYSTEM}, {"role": "user", "content": task}],
        model=model,
        temperature=0.0,
    )
    if tracker is not None:
        tracker.add(resp)
    return _parse_list(resp.text)


@dataclass
class PlanAndExecute:
    """Plan the whole task, then execute each sub-step with a tool-using agent."""

    provider: str = "ollama"
    model: str | None = None
    tools: ToolRegistry | None = None
    tracker: CostTracker = field(default_factory=CostTracker)

    def run(self, task: str) -> AgentResult:
        tracer = Tracer()
        plan = decompose(task, self.provider, self.model, self.tracker)
        for i, step in enumerate(plan, 1):
            tracer.thought(f"Plan step {i}: {step}")
        agent = Agent(
            provider=self.provider,
            model=self.model,
            tools=self.tools if self.tools is not None else default_tools(),
            tracker=self.tracker,
        )
        results: list[str] = []
        for step in plan:
            res = agent.run(step, tracer=tracer)
            results.append(f"- {step} -> {res.answer}")
        # Compose a final answer from the executed steps.
        compose = get_provider(self.provider).chat(
            [
                {"role": "system", "content": "Compose a final answer from the completed steps."},
                {"role": "user", "content": f"Task: {task}\n\nCompleted:\n" + "\n".join(results)},
            ],
            model=self.model,
            temperature=0.0,
        )
        self.tracker.add(compose)
        tracer.answer(compose.text)
        return AgentResult(compose.text, tracer.steps, self.tracker)


def rewoo(
    task: str,
    tools: ToolRegistry | None = None,
    provider: str = "ollama",
    model: str | None = None,
    tracker: Any | None = None,
) -> AgentResult:
    """Plan every tool call up front, execute them, then solve (ReWOO)."""
    tools = tools if tools is not None else default_tools()
    llm = get_provider(provider)
    tracker = tracker if tracker is not None else CostTracker()
    tracer = Tracer()

    plan_prompt = (
        f"Task: {task}\n\nTools: {tools.names}\n\n"
        "List the tool calls needed to solve this, as a JSON array of "
        '{"name": <tool>, "arguments": {...}} objects. No prose.'
    )
    plan_resp = llm.chat(
        [{"role": "system", "content": "You plan tool calls without executing them."},
         {"role": "user", "content": plan_prompt}],
        model=model,
        temperature=0.0,
    )
    tracker.add(plan_resp)

    m = re.search(r"\[.*\]", plan_resp.text, re.DOTALL)
    calls = json.loads(m.group()) if m else []
    evidence = []
    for call in calls:
        name, args = call.get("name", ""), call.get("arguments", {})
        tracer.action(name, args)
        obs = tools.dispatch(name, args)
        tracer.observation(obs, tool=name)
        evidence.append(f"{name}({args}) = {obs}")

    solve = llm.chat(
        [
            {"role": "system", "content": "Answer the task using only the evidence."},
            {"role": "user", "content": f"Task: {task}\n\nEvidence:\n" + "\n".join(evidence)},
        ],
        model=model,
        temperature=0.0,
    )
    tracker.add(solve)
    tracer.answer(solve.text)
    return AgentResult(solve.text, tracer.steps, tracker)
