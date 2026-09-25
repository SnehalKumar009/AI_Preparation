"""orch_lab: the Phase 5 Orchestration core.

Phase 3's agent is a ``while`` loop: the model picks the next step, state lives
in a local ``messages`` list, and everything dies with the process. Orchestration
makes the structure explicit — **state + nodes + edges + checkpoints** — so a
workflow can branch, run steps in parallel, retry, pause for a human, survive a
crash, and be replayed step by step.

As in earlier phases, the package is built **from scratch first** (Part A,
notebooks 01–12) so the mechanics stay visible; later parts map the same ideas
onto LangGraph, the OpenAI Agents SDK, CrewAI, Microsoft Agent Framework,
Google ADK and Temporal.

It reuses Phase 1 ``study_buddy`` (providers + cost tracking), Phase 2
``rag_lab``, Phase 3 ``agents_lab`` (agent loop, tools, tracing) and Phase 4
``mcp_lab`` (the MCP tool servers). Importing ``orch_lab`` locates the sibling
folders that hold those packages and puts them on ``sys.path`` automatically (by
content, not folder name), so notebooks only need to add this package's own root
to the path.

Framework SDKs are never imported here: each framework lives in its own Python
environment (see ``setup.sh``), and :mod:`orch_lab.kernels` tells a notebook
which kernel it needs.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_PKG_DIR = _Path(__file__).resolve().parent      # .../<phase5>/orch_lab
_PHASE5_ROOT = _PKG_DIR.parent                    # .../<phase5>

# Sibling packages this phase reuses, matched by content (folder name-agnostic).
_TARGETS = ("study_buddy", "rag_lab", "agents_lab", "mcp_lab")


def _bootstrap_siblings() -> None:
    """Add sibling folders holding the reused packages to sys.path."""
    found: set[str] = set()
    for base in [_PHASE5_ROOT.parent, *_PHASE5_ROOT.parents]:
        try:
            siblings = [s for s in base.iterdir() if s.is_dir()]
        except (PermissionError, OSError):
            continue
        for sibling in siblings:
            for target in _TARGETS:
                if target not in found and (sibling / target).exists():
                    if str(sibling) not in _sys.path:
                        _sys.path.insert(0, str(sibling))
                    found.add(target)
        if len(found) == len(_TARGETS):
            return
    missing = [t for t in _TARGETS if t not in found]
    if "study_buddy" in missing:
        raise ImportError(
            "Could not locate a sibling folder containing study_buddy. "
            "Phase 5 reuses Phases 1–4; keep them as sibling folders."
        )


_bootstrap_siblings()

from orch_lab.kernels import current_env, requires
from orch_lab.fakes import ScriptedLLM, ProcessCrash

__all__ = [
    "current_env",
    "requires",
    "ScriptedLLM",
    "ProcessCrash",
]
