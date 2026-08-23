"""ReAct: interleave Reasoning and Acting in plain text (notebook 04).

Native tool calling is convenient but hides the model's reasoning. ReAct makes
it explicit with a text protocol the model follows:

    Thought: <reasoning>
    Action: <tool name>
    Action Input: <JSON args>
    Observation: <filled in by us>
    ... (repeat) ...
    Final Answer: <answer>

Because it is just text, it works on any provider with no native tool-calling
support — the same reason ``rag_lab.agentic`` uses a JSON loop. Seeing the
Thought/Action/Observation trace is the whole point.
"""

from __future__ import annotations

import json
import re
from typing import Any

from study_buddy import CostTracker, get_provider

from agents_lab.loop import AgentResult
from agents_lab.tools import ToolRegistry, default_tools
from agents_lab.tracing import Tracer

_SYSTEM = """You solve tasks by reasoning and using tools. Follow this format EXACTLY,
one block at a time, and stop after each Action so the tool can run:

Thought: your reasoning
Action: the tool name, one of: {tool_names}
Action Input: a JSON object of arguments

When you have the answer, instead output:

Thought: your final reasoning
Final Answer: the answer

Available tools:
{tool_descriptions}"""

_ACTION = re.compile(r"Action:\s*(.+?)\s*[\n\r]+Action Input:\s*(\{.*?\}|.+)", re.DOTALL)
_FINAL = re.compile(r"Final Answer:\s*(.+)", re.DOTALL)


def _tool_block(tools: ToolRegistry) -> str:
    return "\n".join(f"- {t.name}: {tools[t.name].description}" for t in [tools[n] for n in tools.names])


def react(
    task: str,
    tools: ToolRegistry | None = None,
    provider: str = "ollama",
    model: str | None = None,
    max_steps: int = 6,
    tracker: CostTracker | None = None,
    tracer: Tracer | None = None,
) -> AgentResult:
    """Run a text-based ReAct loop and return the answer + trajectory."""
    tools = tools if tools is not None else default_tools()
    llm = get_provider(provider)
    tracker = tracker if tracker is not None else CostTracker()
    tracer = tracer or Tracer()

    system = _SYSTEM.format(tool_names=", ".join(tools.names), tool_descriptions=_tool_block(tools))
    transcript = f"Task: {task}\n"

    for _ in range(max_steps):
        resp = llm.chat(
            [{"role": "system", "content": system}, {"role": "user", "content": transcript}],
            model=model,
            temperature=0.0,
        )
        tracker.add(resp)
        text = resp.text

        final = _FINAL.search(text)
        if final:
            answer = final.group(1).strip()
            tracer.answer(answer)
            return AgentResult(answer, tracer.steps, tracker)

        m = _ACTION.search(text)
        if not m:
            # No parsable action: treat the whole reply as the answer.
            tracer.answer(text.strip())
            return AgentResult(text.strip(), tracer.steps, tracker)

        thought = text[: m.start()].replace("Thought:", "").strip()
        if thought:
            tracer.thought(thought)
        name = m.group(1).strip()
        raw_args = m.group(2).strip()
        try:
            args = json.loads(raw_args)
        except json.JSONDecodeError:
            args = {"input": raw_args}

        tracer.action(name, args)
        obs = tools.dispatch(name, args)
        tracer.observation(obs, tool=name)
        transcript += f"{text.strip()}\nObservation: {obs}\n"

    tracer.answer("(max steps reached)")
    return AgentResult("(max steps reached)", tracer.steps, tracker)
