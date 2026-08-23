"""agents_lab: the Phase 3 AI-Agents core.

Reuses Phase 1 ``study_buddy`` (providers + cost tracking) and Phase 2
``rag_lab`` (retrieval), and adds agent machinery: the agent loop, a tool
registry, ReAct, planning, reflection/Reflexion, self-correction, loop control,
short-term memory, safety, agentic RAG, light multi-agent patterns, tracing, and
a framework bridge. Importing ``agents_lab`` locates the sibling folders that
hold ``study_buddy`` and ``rag_lab`` and puts them on ``sys.path`` automatically
(by content, not folder name), so notebooks only need to add this package's own
root to the path.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_PKG_DIR = _Path(__file__).resolve().parent      # .../<phase3>/agents_lab
_PHASE3_ROOT = _PKG_DIR.parent                    # .../<phase3>


def _bootstrap_siblings() -> None:
    """Add sibling folders holding study_buddy and rag_lab to sys.path (name-agnostic)."""
    found1 = found2 = False
    for base in [_PHASE3_ROOT.parent, *_PHASE3_ROOT.parents]:
        try:
            siblings = [s for s in base.iterdir() if s.is_dir()]
        except (PermissionError, OSError):
            continue
        for sibling in siblings:
            if not found1 and (sibling / "study_buddy").exists():
                if str(sibling) not in _sys.path:
                    _sys.path.insert(0, str(sibling))
                found1 = True
            if not found2 and (sibling / "rag_lab").exists():
                if str(sibling) not in _sys.path:
                    _sys.path.insert(0, str(sibling))
                found2 = True
        if found1 and found2:
            return
    if not found1:
        raise ImportError(
            "Could not locate a sibling folder containing study_buddy."
        )


_bootstrap_siblings()

from agents_lab.tracing import Step, Tracer
from agents_lab.tools import Tool, ToolRegistry, default_tools
from agents_lab.loop import Agent, AgentResult, DEFAULT_SYSTEM
from agents_lab.react import react
from agents_lab.reasoning import chain_of_thought, self_consistency
from agents_lab.planning import decompose, PlanAndExecute, rewoo
from agents_lab.tot import tree_of_thoughts, ThoughtNode
from agents_lab.reflection import reflect, reflect_and_revise
from agents_lab.reflexion import Reflexion, reflexion
from agents_lab.correction import self_correct, repair_json, with_retry
from agents_lab.control import Budget, Guard, BudgetExceeded, RunawayLoop
from agents_lab.memory import Scratchpad, ShortTermMemory
from agents_lab.safety import check_tool_input, guard_observation, SafetyReport
from agents_lab.agentic_rag import make_rag_tool, build_rag_agent
from agents_lab.multiagent import Worker, Supervisor, debate
from agents_lab import frameworks

__all__ = [
    "Step",
    "Tracer",
    "Tool",
    "ToolRegistry",
    "default_tools",
    "Agent",
    "AgentResult",
    "DEFAULT_SYSTEM",
    "react",
    "chain_of_thought",
    "self_consistency",
    "decompose",
    "PlanAndExecute",
    "rewoo",
    "tree_of_thoughts",
    "ThoughtNode",
    "reflect",
    "reflect_and_revise",
    "Reflexion",
    "reflexion",
    "self_correct",
    "repair_json",
    "with_retry",
    "Budget",
    "Guard",
    "BudgetExceeded",
    "RunawayLoop",
    "Scratchpad",
    "ShortTermMemory",
    "check_tool_input",
    "guard_observation",
    "SafetyReport",
    "make_rag_tool",
    "build_rag_agent",
    "Worker",
    "Supervisor",
    "debate",
    "frameworks",
]
