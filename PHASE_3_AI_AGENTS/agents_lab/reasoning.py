"""Reasoning patterns: Chain-of-Thought and Self-Consistency (notebook 05).

- ``chain_of_thought`` asks the model to reason step by step before answering,
  which improves multi-step arithmetic and logic.
- ``self_consistency`` samples several independent reasoning paths (temperature
  > 0) and takes a majority vote on the final answer, trading cost for accuracy.

These are the reasoning primitives the planning and tree-search modules build on.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

from study_buddy import CostTracker, get_provider

_COT_SYSTEM = (
    "Reason step by step. Show your working, then end with a line of the exact form "
    "'Answer: <result>' containing only the final result."
)


def _extract_answer(text: str) -> str:
    m = re.search(r"Answer:\s*(.+)", text, re.IGNORECASE)
    return (m.group(1) if m else text).strip().splitlines()[0].strip()


def chain_of_thought(
    question: str,
    provider: str = "ollama",
    model: str | None = None,
    tracker: Any | None = None,
) -> tuple[str, str]:
    """Return (final_answer, full_reasoning)."""
    resp = get_provider(provider).chat(
        [{"role": "system", "content": _COT_SYSTEM}, {"role": "user", "content": question}],
        model=model,
        temperature=0.0,
    )
    if tracker is not None:
        tracker.add(resp)
    return _extract_answer(resp.text), resp.text


def self_consistency(
    question: str,
    n: int = 5,
    provider: str = "ollama",
    model: str | None = None,
    temperature: float = 0.8,
    tracker: Any | None = None,
) -> tuple[str, list[str]]:
    """Sample ``n`` reasoning paths and majority-vote the answer.

    Returns (voted_answer, all_sampled_answers).
    """
    llm = get_provider(provider)
    tracker = tracker if tracker is not None else CostTracker()
    answers: list[str] = []
    for _ in range(n):
        resp = llm.chat(
            [{"role": "system", "content": _COT_SYSTEM}, {"role": "user", "content": question}],
            model=model,
            temperature=temperature,
        )
        tracker.add(resp)
        answers.append(_extract_answer(resp.text))
    voted = Counter(answers).most_common(1)[0][0]
    return voted, answers
