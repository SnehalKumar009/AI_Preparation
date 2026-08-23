"""CostTracker: accumulate USD, tokens, and latency across calls."""

from __future__ import annotations

from dataclasses import dataclass, field

from study_buddy.pricing import estimate_cost
from study_buddy.providers.base import ChatResponse


@dataclass
class CallRecord:
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_s: float


@dataclass
class CostTracker:
    """Log each call and report running totals. Ollama calls cost $0."""

    records: list[CallRecord] = field(default_factory=list)

    def add(self, resp: ChatResponse) -> CallRecord:
        rec = CallRecord(
            provider=resp.provider,
            model=resp.model,
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
            cost_usd=estimate_cost(resp.model, resp.usage),
            latency_s=resp.latency_s,
        )
        self.records.append(rec)
        return rec

    @property
    def total_cost(self) -> float:
        return sum(r.cost_usd for r in self.records)

    @property
    def total_tokens(self) -> int:
        return sum(r.input_tokens + r.output_tokens for r in self.records)

    def summary(self) -> str:
        return (
            f"{len(self.records)} calls | "
            f"{self.total_tokens} tokens | ${self.total_cost:.6f}"
        )

    def table(self) -> str:
        """Plain-text table of every call (handy in notebooks and Streamlit)."""
        head = f"{'provider':<10} {'model':<22} {'in':>7} {'out':>7} {'USD':>10} {'sec':>7}"
        lines = [head, "-" * len(head)]
        for r in self.records:
            lines.append(
                f"{r.provider:<10} {r.model:<22} {r.input_tokens:>7} "
                f"{r.output_tokens:>7} {r.cost_usd:>10.6f} {r.latency_s:>7.2f}"
            )
        lines.append("-" * len(head))
        lines.append(f"{'TOTAL':<10} {'':<22} {'':>7} {'':>7} {self.total_cost:>10.6f}")
        return "\n".join(lines)
