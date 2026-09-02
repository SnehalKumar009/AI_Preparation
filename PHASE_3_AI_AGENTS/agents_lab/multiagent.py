"""Multi-agent basics: supervisor, handoff, debate, pipeline, group chat (16-17, 20-21).

One agent is often not enough. This module shows the light-weight coordination
patterns; deep orchestration (state machines, checkpointing, durable workflows)
is Phase 5.

- :class:`Worker` — a named specialist backed by an :class:`~agents_lab.loop.Agent`.
- :class:`Supervisor` — routes a task to the most suitable worker (delegation).
- ``debate`` — two personas argue and a judge picks the stronger answer.
- :class:`Pipeline` — an ordered chain of workers, each consuming the previous
  one's output (e.g. Planner -> Research -> Coder -> Reviewer -> Tester).
- :class:`GroupChat` — several workers take turns over a shared transcript.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from study_buddy import CostTracker, get_provider

from agents_lab.loop import Agent
from agents_lab.tools import ToolRegistry


@dataclass
class Worker:
    name: str
    description: str
    agent: Agent

    def run(self, task: str) -> str:
        return self.agent.run(task).answer


class Supervisor:
    """Route tasks to named workers (handoff / delegation)."""

    def __init__(
        self,
        workers: list[Worker],
        provider: str = "ollama",
        model: str | None = None,
        tracker: CostTracker | None = None,
    ) -> None:
        self.workers = {w.name: w for w in workers}
        self.provider = provider
        self.model = model
        self.tracker = tracker if tracker is not None else CostTracker()

    def route(self, task: str) -> str:
        """Pick the best worker name for a task."""
        roster = "\n".join(f"- {w.name}: {w.description}" for w in self.workers.values())
        resp = get_provider(self.provider).chat(
            [
                {"role": "system", "content": "Choose the single best worker for the task. Reply with only their name."},
                {"role": "user", "content": f"Task: {task}\n\nWorkers:\n{roster}"},
            ],
            model=self.model,
            temperature=0.0,
        )
        self.tracker.add(resp)
        choice = resp.text.strip().split()[0] if resp.text.strip() else ""
        return choice if choice in self.workers else next(iter(self.workers))

    def run(self, task: str) -> tuple[str, str]:
        """Route then delegate. Returns (worker_name, answer)."""
        name = self.route(task)
        return name, self.workers[name].run(task)


def debate(
    question: str,
    provider: str = "ollama",
    model: str | None = None,
    rounds: int = 2,
    personas: tuple[str, str] = ("Proponent", "Skeptic"),
    tracker: Any | None = None,
) -> str:
    """Two personas argue for a few rounds; a judge returns the final verdict."""
    llm = get_provider(provider)
    tracker = tracker if tracker is not None else CostTracker()
    transcript = f"Question: {question}\n"
    for _ in range(rounds):
        for persona in personas:
            resp = llm.chat(
                [
                    {"role": "system", "content": f"You are the {persona}. Argue your side concisely and rebut the other."},
                    {"role": "user", "content": transcript},
                ],
                model=model,
                temperature=0.5,
            )
            tracker.add(resp)
            transcript += f"\n{persona}: {resp.text}\n"
    judge = llm.chat(
        [
            {"role": "system", "content": "You are the judge. Weigh both sides and give the best-supported final answer."},
            {"role": "user", "content": transcript},
        ],
        model=model,
        temperature=0.0,
    )
    tracker.add(judge)
    return judge.text


@dataclass
class PipelineResult:
    final: str
    outputs: list[tuple[str, str]] = field(default_factory=list)  # (stage, output)

    def pretty(self) -> str:
        return "\n\n".join(f"## {name}\n{text}" for name, text in self.outputs)


class Pipeline:
    """An ordered chain of workers; each stage consumes the previous output.

    This is the classic assembly-line topology, e.g.
    Planner -> Research -> Coder -> Reviewer -> Tester. Each worker sees the
    original task plus the prior stage's output, so responsibilities stay narrow.
    """

    def __init__(self, stages: list[Worker]) -> None:
        if not stages:
            raise ValueError("Pipeline needs at least one stage")
        self.stages = stages

    def run(self, task: str) -> PipelineResult:
        outputs: list[tuple[str, str]] = []
        prior = ""
        for worker in self.stages:
            prompt = task if not prior else (
                f"Task: {task}\n\nOutput from the previous stage "
                f"({outputs[-1][0]}):\n{prior}\n\nDo your part."
            )
            answer = worker.run(prompt)
            outputs.append((worker.name, answer))
            prior = answer
        return PipelineResult(final=prior, outputs=outputs)


class GroupChat:
    """Several workers take turns contributing to one shared transcript.

    Unlike a pipeline (one pass, fixed order), a group chat loops for a few
    rounds so agents can build on and correct each other. An optional final
    summary distills the discussion into one answer.
    """

    def __init__(
        self,
        members: list[Worker],
        rounds: int = 2,
        provider: str = "ollama",
        model: str | None = None,
        tracker: CostTracker | None = None,
    ) -> None:
        if not members:
            raise ValueError("GroupChat needs at least one member")
        self.members = members
        self.rounds = rounds
        self.provider = provider
        self.model = model
        self.tracker = tracker if tracker is not None else CostTracker()

    def run(self, task: str, summarize: bool = True) -> str:
        transcript = f"Topic: {task}\n"
        for _ in range(self.rounds):
            for member in self.members:
                turn = member.run(
                    f"You are {member.name} ({member.description}). "
                    f"Contribute to the discussion; build on or correct others.\n\n{transcript}"
                )
                transcript += f"\n{member.name}: {turn}\n"
        if not summarize:
            return transcript
        summary = get_provider(self.provider).chat(
            [
                {"role": "system", "content": "Summarize the discussion into one clear, final answer."},
                {"role": "user", "content": transcript},
            ],
            model=self.model,
            temperature=0.0,
        )
        self.tracker.add(summary)
        return summary.text
