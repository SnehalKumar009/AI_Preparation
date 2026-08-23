"""Loop control: keep agents from running away (notebook 12).

An agent loop can spin forever or quietly burn money. A :class:`Budget` sets hard
limits on steps, USD, and tokens; a :class:`Guard` checks them against a live
``study_buddy.CostTracker`` and raises before the next expensive call. This is the
safety rail every long-running agent needs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class BudgetExceeded(RuntimeError):
    """Raised when an agent exceeds its cost or token budget."""


class RunawayLoop(RuntimeError):
    """Raised when an agent exceeds its step budget."""


@dataclass
class Budget:
    max_steps: int | None = 10
    max_usd: float | None = None
    max_tokens: int | None = None


class Guard:
    """Enforce a :class:`Budget` against a CostTracker over a run."""

    def __init__(self, budget: Budget) -> None:
        self.budget = budget
        self.steps = 0

    def tick(self, tracker: Any | None = None) -> None:
        """Call once per loop iteration; raises if any limit is crossed."""
        self.steps += 1
        b = self.budget
        if b.max_steps is not None and self.steps > b.max_steps:
            raise RunawayLoop(f"exceeded {b.max_steps} steps")
        if tracker is not None:
            if b.max_usd is not None and tracker.total_cost > b.max_usd:
                raise BudgetExceeded(f"exceeded ${b.max_usd:.4f} (spent ${tracker.total_cost:.4f})")
            if b.max_tokens is not None and tracker.total_tokens > b.max_tokens:
                raise BudgetExceeded(f"exceeded {b.max_tokens} tokens")

    def remaining_steps(self) -> int | None:
        if self.budget.max_steps is None:
            return None
        return max(0, self.budget.max_steps - self.steps)
