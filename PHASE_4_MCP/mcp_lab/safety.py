"""Safety primitives for local MCP servers (notebooks 23–26).

Local MCP servers touch the real filesystem, git, a database, and the shell, so
they need guards that a doc-RAG pipeline does not. This module adds MCP-specific
defenses and re-exposes Phase 2's heuristics so there is one place to reason
about safety:

- ``resolve_within_root`` — block path traversal; keep file access under a root.
- ``is_allowed_command`` / ``split_command`` — allowlist shell commands.
- ``redact_pii`` / ``detect_injection`` — reused from ``rag_lab.guardrails`` so
  tool output can be scrubbed before it reaches the model.
"""

from __future__ import annotations

import shlex
from pathlib import Path

__all__ = [
    "SandboxError",
    "resolve_within_root",
    "split_command",
    "is_allowed_command",
    "redact_pii",
    "detect_injection",
]


class SandboxError(Exception):
    """Raised when an operation would escape its allowed sandbox."""


def resolve_within_root(root: str | Path, candidate: str | Path) -> Path:
    """Resolve ``candidate`` and guarantee it stays inside ``root``.

    Defeats ``../`` traversal and absolute-path escapes by comparing the fully
    resolved paths. Raises :class:`SandboxError` on any escape attempt.
    """
    root_path = Path(root).resolve()
    target = (root_path / candidate).resolve() if not Path(candidate).is_absolute() else Path(candidate).resolve()
    if root_path != target and root_path not in target.parents:
        raise SandboxError(f"Path escapes sandbox root: {candidate}")
    return target


def split_command(command: str) -> list[str]:
    """Tokenize a shell command without invoking a shell (no injection)."""
    return shlex.split(command)


def is_allowed_command(command: str, allowlist: set[str] | frozenset[str]) -> bool:
    """True only if the command's program (argv[0]) is in the allowlist."""
    parts = split_command(command)
    if not parts:
        return False
    return parts[0] in allowlist


# ---- Reused Phase 2 guardrails (imported lazily to keep the core light) ----
def redact_pii(text: str) -> tuple[str, list[str]]:
    """Mask PII in tool output. Reuses ``rag_lab.guardrails.redact_pii``."""
    from rag_lab.guardrails import redact_pii as _redact

    return _redact(text)


def detect_injection(text: str) -> bool:
    """Flag prompt-injection phrases. Reuses ``rag_lab.guardrails.detect_injection``."""
    from rag_lab.guardrails import detect_injection as _detect

    return _detect(text)
