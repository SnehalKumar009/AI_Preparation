"""Security & guardrails (notebook 26).

RAG widens the attack surface: retrieved documents become part of the prompt, so a
poisoned corpus can carry **prompt-injection** instructions, and answers can leak
**PII**. This module provides fast, dependency-free defenses:

- ``detect_injection`` — flag imperative override phrases in retrieved text.
- ``redact_pii``       — mask emails, phone numbers, SSN/card-like digit runs.
- ``sanitize_hits``    — redact + drop injected chunks before they reach the model.
- ``is_grounded``      — cheap check that the answer overlaps the provided context.

These are heuristics — a first layer, not a guarantee. An optional LLM classifier
can back them up.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from rag_lab.vectorstore import Hit

_INJECTION = re.compile(
    r"\b(ignore (all |the )?(previous|above)|disregard (the |all )?(previous|instructions)|"
    r"forget (everything|your instructions)|you are now|new instructions?:|"
    r"system prompt|reveal (your |the )?(prompt|instructions)|do anything now)\b",
    re.IGNORECASE,
)

_PII = {
    "email": re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"),
    "phone": re.compile(r"\b(?:\+?\d{1,2}[\s-]?)?(?:\(?\d{3}\)?[\s-]?)\d{3}[\s-]?\d{4}\b"),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "card": re.compile(r"\b(?:\d[ -]?){13,16}\b"),
}


@dataclass
class GuardReport:
    injection: bool = False
    pii_found: list[str] = field(default_factory=list)


def detect_injection(text: str) -> bool:
    """True if the text contains a known prompt-injection phrase."""
    return bool(_INJECTION.search(text))


def redact_pii(text: str) -> tuple[str, list[str]]:
    """Mask PII; return the cleaned text and the list of categories found."""
    found: list[str] = []
    for label, pattern in _PII.items():
        if pattern.search(text):
            found.append(label)
            text = pattern.sub(f"[{label.upper()}_REDACTED]", text)
    return text, found


def scan(text: str) -> GuardReport:
    """Run all input checks and report what was found."""
    _, pii = redact_pii(text)
    return GuardReport(injection=detect_injection(text), pii_found=pii)


def sanitize_hits(hits: list[Hit], drop_injected: bool = True) -> tuple[list[Hit], list[GuardReport]]:
    """Redact PII in each hit and optionally drop chunks containing injection."""
    clean: list[Hit] = []
    reports: list[GuardReport] = []
    for h in hits:
        report = scan(h.chunk.text)
        reports.append(report)
        if drop_injected and report.injection:
            continue
        safe_text, _ = redact_pii(h.chunk.text)
        chunk = h.chunk
        clean.append(Hit(type(chunk)(safe_text, chunk.source, chunk.index, dict(chunk.meta)), h.score))
    return clean, reports


def is_grounded(answer: str, context: str, min_overlap: float = 0.15) -> bool:
    """Cheap grounding check: share of answer words also present in the context."""
    ans_words = {w.lower() for w in re.findall(r"\w+", answer) if len(w) > 3}
    ctx_words = {w.lower() for w in re.findall(r"\w+", context)}
    if not ans_words:
        return False
    return len(ans_words & ctx_words) / len(ans_words) >= min_overlap
