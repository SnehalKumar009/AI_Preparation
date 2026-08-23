"""RAG evaluation (notebook 21).

You cannot improve what you do not measure. These are LLM-as-judge implementations
of the metrics popularized by RAGAS, computed by hand so the mechanics are visible:

- **context_precision** — fraction of retrieved chunks that are actually relevant.
- **context_recall**    — fraction of the ground-truth answer supported by the context.
- **faithfulness**      — is the generated answer entailed by the context (not made up)?
- **answer_relevance**  — does the answer actually address the question?

Each returns a 0–1 score. A single judge call per metric keeps cost low, so any
provider works; token budgets are tiny.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from study_buddy import get_provider

from rag_lab.vectorstore import Hit

_NUM = re.compile(r"\d+(?:\.\d+)?")

_YESNO = (
    "Answer with only YES or NO.\n\n{question}\n\n"
    "STATEMENT: {statement}\n\nCONTEXT:\n{context}"
)
_RELEVANCE = (
    "On a scale of 0 to 10, how directly does the ANSWER address the QUESTION? "
    "Reply with only the number.\n\nQUESTION: {question}\n\nANSWER: {answer}\n\nScore:"
)


@dataclass
class EvalResult:
    context_precision: float = 0.0
    context_recall: float = 0.0
    faithfulness: float = 0.0
    answer_relevance: float = 0.0

    def summary(self) -> str:
        return (
            f"precision={self.context_precision:.2f} recall={self.context_recall:.2f} "
            f"faithfulness={self.faithfulness:.2f} relevance={self.answer_relevance:.2f}"
        )


class Judge:
    """Thin LLM wrapper returning yes/no and 0–1 scores for eval prompts."""

    def __init__(self, provider: str = "ollama", model: str | None = None, tracker: Any | None = None):
        self.provider = provider
        self.model = model
        self.tracker = tracker
        self._llm = None

    def _ask(self, prompt: str) -> str:
        if self._llm is None:
            self._llm = get_provider(self.provider)
        resp = self._llm.chat([{"role": "user", "content": prompt}], model=self.model, temperature=0.0)
        if self.tracker is not None:
            self.tracker.add(resp)
        return resp.text.strip()

    def _yes(self, question: str, statement: str, context: str) -> bool:
        print("==========JUDGE YES/NO START==========")
        print("question:", question)
        print("statement:", statement)
        print("context:", context)
        
        out = self._ask(_YESNO.format(question=question, statement=statement, context=context))
        print("out:", out)
        print("==========JUDGE YES/NO END==========")
        return out.upper().lstrip().startswith("Y")

    def _score01(self, prompt: str) -> float:
        m = _NUM.search(self._ask(prompt))
        return min(1.0, float(m.group()) / 10.0) if m else 0.0


def context_precision(judge: Judge, question: str, hits: list[Hit]) -> float:
    """Share of retrieved chunks the judge deems relevant to the question."""
    if not hits:
        return 0.0
    q = "Is this CONTEXT relevant to answering the QUESTION below?\nQUESTION: " + question
    print("==========CONTEXT PRECISION START==========")
    relevant = sum(judge._yes(q, "", h.chunk.text) for h in hits)
    print("==========CONTEXT PRECISION END==========")
    return relevant / len(hits)


def context_recall(judge: Judge, ground_truth: str, hits: list[Hit]) -> float:
    """Share of ground-truth sentences supported by the retrieved context."""
    from rag_lab.chunking import _sentences
    print("==========CONTEXT RECALL START==========")
    sents = _sentences(ground_truth)
    if not sents:
        return 0.0
    context = "\n\n".join(h.chunk.text for h in hits)
    q = "Is the STATEMENT supported by the CONTEXT?"
    supported = sum(judge._yes(q, s, context) for s in sents)
    print("==========CONTEXT RECALL END==========")
    return supported / len(sents)


def faithfulness(judge: Judge, answer: str, hits: list[Hit]) -> float:
    """Share of the answer's sentences entailed by the retrieved context."""
    from rag_lab.chunking import _sentences
    print("==========FAITHFULNESS START==========")
    sents = _sentences(answer)
    if not sents:
        return 0.0
    context = "\n\n".join(h.chunk.text for h in hits)
    q = "Is the STATEMENT fully supported by the CONTEXT (no invented facts)?"
    grounded = sum(judge._yes(q, s, context) for s in sents)
    print("==========FAITHFULNESS END==========")
    return grounded / len(sents)


def answer_relevance(judge: Judge, question: str, answer: str) -> float:
    """How directly the answer addresses the question (0–1)."""
    return judge._score01(_RELEVANCE.format(question=question, answer=answer))


def evaluate(
    question: str,
    answer: str,
    hits: list[Hit],
    ground_truth: str | None = None,
    provider: str = "ollama",
    model: str | None = None,
    tracker: Any | None = None,
) -> EvalResult:
    """Run the full metric suite for one (question, answer, context) example."""
    print("==========EVALUATION START==========")
    print("question:", question)
    print("answer:", answer)
    print("ground_truth:", ground_truth)
    print("hits:")
    for hit in hits:
        print("hit.chunk.source:", hit.chunk.source)
        print("hit.chunk.text:", hit.chunk.text)
    print("==========EVALUATION END==========")
    judge = Judge(provider=provider, model=model, tracker=tracker)
    return EvalResult(
        context_precision=context_precision(judge, question, hits),
        context_recall=context_recall(judge, ground_truth, hits) if ground_truth else 0.0,
        faithfulness=faithfulness(judge, answer, hits),
        answer_relevance=answer_relevance(judge, question, answer),
    )
