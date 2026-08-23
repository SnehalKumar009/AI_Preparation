"""Reflexion: reflection with memory across attempts (notebook 10).

Plain reflection critiques one answer. Reflexion (Shinn et al., 2023) adds an
*episodic memory of lessons*: after each failed attempt an evaluator gives
feedback, the agent writes a short self-reflection, and that reflection is
carried into the next attempt so it doesn't repeat the mistake. It needs an
``evaluator(answer) -> (passed, feedback)`` that judges success.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from study_buddy import CostTracker, get_provider

from agents_lab.loop import AgentResult
from agents_lab.tracing import Tracer

Evaluator = Callable[[str], "tuple[bool, str]"]


@dataclass
class Reflexion:
    """Retry a task, accumulating self-reflections between trials."""

    provider: str = "ollama"
    model: str | None = None
    max_trials: int = 3
    tracker: CostTracker = field(default_factory=CostTracker)
    reflections: list[str] = field(default_factory=list)

    def _attempt(self, task: str) -> str:
        memory = "\n".join(f"- {r}" for r in self.reflections) or "(none)"
        resp = get_provider(self.provider).chat(
            [
                {"role": "system", "content": "Solve the task. Apply lessons from past reflections."},
                {"role": "user", "content": f"Task: {task}\n\nLessons from past attempts:\n{memory}"},
            ],
            model=self.model,
            temperature=0.3,
        )
        self.tracker.add(resp)
        return resp.text

    def _reflect(self, task: str, answer: str, feedback: str) -> str:
        resp = get_provider(self.provider).chat(
            [
                {"role": "system", "content": "In one sentence, state the lesson to avoid this mistake next time."},
                {"role": "user", "content": f"Task: {task}\nAnswer: {answer}\nFeedback: {feedback}"},
            ],
            model=self.model,
            temperature=0.3,
        )
        self.tracker.add(resp)
        return resp.text.strip()

    def run(self, task: str, evaluator: Evaluator) -> AgentResult:
        tracer = Tracer()
        answer = ""
        for trial in range(1, self.max_trials + 1):
            answer = self._attempt(task)
            tracer.answer(answer, trial=trial)
            passed, feedback = evaluator(answer)
            tracer.observation(f"eval: {'PASS' if passed else 'FAIL'} - {feedback}", trial=trial)
            if passed:
                break
            lesson = self._reflect(task, answer, feedback)
            self.reflections.append(lesson)
            tracer.thought(f"reflection: {lesson}", trial=trial)
        return AgentResult(answer, tracer.steps, self.tracker)


def reflexion(
    task: str,
    evaluator: Evaluator,
    provider: str = "ollama",
    model: str | None = None,
    max_trials: int = 3,
    tracker: Any | None = None,
) -> AgentResult:
    """Functional wrapper around :class:`Reflexion`."""
    engine = Reflexion(
        provider=provider,
        model=model,
        max_trials=max_trials,
        tracker=tracker if tracker is not None else CostTracker(),
    )
    return engine.run(task, evaluator)
