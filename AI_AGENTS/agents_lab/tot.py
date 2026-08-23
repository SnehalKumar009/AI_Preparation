"""Tree of Thoughts: search over reasoning branches (notebook 08).

Chain-of-Thought commits to a single line of reasoning. Tree of Thoughts (ToT)
instead *expands* several candidate next-thoughts at each level, *evaluates* how
promising each one is, keeps the best few (beam search), and continues to a set
depth. It trades many more model calls for better results on problems where a
single greedy path easily goes wrong.

This is a compact, teaching implementation: expand -> score -> keep top-k.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from study_buddy import CostTracker, get_provider


@dataclass
class ThoughtNode:
    text: str
    score: float = 0.0
    path: list[str] = field(default_factory=list)


def _expand(llm, problem: str, path: list[str], breadth: int, model, tracker) -> list[str]:
    context = "\n".join(f"- {p}" for p in path) or "(none yet)"
    resp = llm.chat(
        [
            {"role": "system", "content": "Propose distinct next reasoning steps toward the solution."},
            {"role": "user", "content":
                f"Problem: {problem}\n\nSteps so far:\n{context}\n\n"
                f"Give {breadth} different possible NEXT steps, one per line, no numbering."},
        ],
        model=model,
        temperature=0.9,
    )
    tracker.add(resp)
    return [ln.strip("-* ").strip() for ln in resp.text.splitlines() if ln.strip()][:breadth]


def _score(llm, problem: str, path: list[str], candidate: str, model, tracker) -> float:
    resp = llm.chat(
        [
            {"role": "system", "content": "Rate from 0 to 10 how promising a reasoning step is. Reply with only the number."},
            {"role": "user", "content": f"Problem: {problem}\nStep: {candidate}"},
        ],
        model=model,
        temperature=0.0,
    )
    tracker.add(resp)
    m = re.search(r"\d+(\.\d+)?", resp.text)
    return float(m.group()) if m else 0.0


def tree_of_thoughts(
    problem: str,
    provider: str = "ollama",
    model: str | None = None,
    breadth: int = 3,
    depth: int = 2,
    beam: int = 2,
    tracker: Any | None = None,
) -> tuple[str, list[ThoughtNode]]:
    """Beam-search over thoughts. Returns (final_answer, best_path_nodes)."""
    llm = get_provider(provider)
    tracker = tracker if tracker is not None else CostTracker()

    frontier = [ThoughtNode(text="start", path=[])]
    for _ in range(depth):
        candidates: list[ThoughtNode] = []
        for node in frontier:
            for cand in _expand(llm, problem, node.path, breadth, model, tracker):
                score = _score(llm, problem, node.path, cand, model, tracker)
                candidates.append(ThoughtNode(cand, score, node.path + [cand]))
        if not candidates:
            break
        frontier = sorted(candidates, key=lambda n: n.score, reverse=True)[:beam]

    best_path = frontier[0].path if frontier else []
    solve = llm.chat(
        [
            {"role": "system", "content": "Give the final answer using the reasoning path."},
            {"role": "user", "content": f"Problem: {problem}\n\nReasoning:\n" + "\n".join(best_path)},
        ],
        model=model,
        temperature=0.0,
    )
    tracker.add(solve)
    return solve.text, frontier
