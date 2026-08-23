"""Reflection / self-critique: the agent grades its own work (notebook 09).

A first draft is rarely the best answer. Reflection adds a critic step: the model
reviews its own output, names concrete problems, then revises. ``reflect`` runs a
single critique; ``reflect_and_revise`` loops draft -> critique -> revise until
the critic is satisfied or a round cap is hit.
"""

from __future__ import annotations

from typing import Any

from study_buddy import CostTracker, get_provider

from agents_lab.loop import AgentResult
from agents_lab.tracing import Tracer

_CRITIC = (
    "You are a strict critic. Review the answer for correctness, completeness, and "
    "clarity. List concrete problems as short bullets. If the answer is fully correct "
    "and complete, reply with exactly 'OK'."
)


def reflect(
    task: str,
    answer: str,
    provider: str = "ollama",
    model: str | None = None,
    tracker: Any | None = None,
) -> str:
    """Return a critique of ``answer`` (or 'OK' if it passes)."""
    resp = get_provider(provider).chat(
        [
            {"role": "system", "content": _CRITIC},
            {"role": "user", "content": f"Task: {task}\n\nAnswer:\n{answer}"},
        ],
        model=model,
        temperature=0.0,
    )
    if tracker is not None:
        tracker.add(resp)
    return resp.text.strip()


def reflect_and_revise(
    task: str,
    provider: str = "ollama",
    model: str | None = None,
    max_rounds: int = 2,
    tracker: Any | None = None,
) -> AgentResult:
    """Draft, then critique-and-revise until the critic says 'OK'."""
    llm = get_provider(provider)
    tracker = tracker if tracker is not None else CostTracker()
    tracer = Tracer()

    draft_resp = llm.chat(
        [{"role": "system", "content": "Answer the task."}, {"role": "user", "content": task}],
        model=model,
        temperature=0.2,
    )
    tracker.add(draft_resp)
    answer = draft_resp.text
    tracer.answer(answer, stage="draft")

    for _ in range(max_rounds):
        critique = reflect(task, answer, provider, model, tracker)
        tracer.thought(critique, stage="critique")
        if critique.strip().upper().startswith("OK"):
            break
        revise = llm.chat(
            [
                {"role": "system", "content": "Revise the answer to address every critique point."},
                {"role": "user", "content": f"Task: {task}\n\nAnswer:\n{answer}\n\nCritique:\n{critique}"},
            ],
            model=model,
            temperature=0.2,
        )
        tracker.add(revise)
        answer = revise.text
        tracer.answer(answer, stage="revised")

    return AgentResult(answer, tracer.steps, tracker)
