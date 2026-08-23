"""Agent safety: guard tool inputs and observations (notebook 14).

Tools widen the attack surface: a web page or retrieved document an agent reads
can carry **prompt-injection** ("ignore your instructions and..."), and tool
outputs can leak **PII**. This module reuses Phase 2's ``rag_lab.guardrails``
heuristics and adds agent-facing helpers so injected instructions are flagged and
PII is redacted before text re-enters the model's context. Deeper production
security is Phase 8 — this is the first line of defense.
"""

from __future__ import annotations

from dataclasses import dataclass

from rag_lab.guardrails import detect_injection, redact_pii


@dataclass
class SafetyReport:
    injection: bool = False
    pii_found: list[str] | None = None
    safe_text: str = ""


def check_tool_input(text: str) -> SafetyReport:
    """Flag prompt-injection and redact PII in text before the agent reads it."""
    injection = detect_injection(text)
    safe_text, pii = redact_pii(text)
    return SafetyReport(injection=injection, pii_found=pii, safe_text=safe_text)


def guard_observation(text: str, block_injection: bool = True) -> str:
    """Return a sanitized observation; neutralize injected instructions."""
    report = check_tool_input(text)
    if block_injection and report.injection:
        return (
            "[blocked: the tool output contained instructions that were ignored] "
            + report.safe_text
        )
    return report.safe_text
