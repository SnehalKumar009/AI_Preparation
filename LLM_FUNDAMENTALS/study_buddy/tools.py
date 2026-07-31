"""Example tools for function calling (topic 10 + capstone).

Each tool has an OpenAI-style JSON schema (also accepted by Ollama) plus a
Python implementation. `dispatch` runs a tool call the model requested.
"""

from __future__ import annotations

import ast
import datetime as _dt
import operator as _op
from typing import Any, Callable

# ---- Implementations -------------------------------------------------------

# Safe arithmetic: evaluate a math expression without exec/eval on arbitrary code.
_OPS = {
    ast.Add: _op.add,
    ast.Sub: _op.sub,
    ast.Mult: _op.mul,
    ast.Div: _op.truediv,
    ast.Pow: _op.pow,
    ast.USub: _op.neg,
    ast.Mod: _op.mod,
}


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp):
        return _OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp):
        return _OPS[type(node.op)](_eval_node(node.operand))
    raise ValueError("unsupported expression")


def calculator(expression: str) -> str:
    return str(_eval_node(ast.parse(expression, mode="eval").body))


def current_time(timezone: str = "local") -> str:
    return _dt.datetime.now().isoformat(timespec="seconds")


# ---- Schemas (sent to the model) ------------------------------------------

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a basic arithmetic expression, e.g. '2 * (3 + 4)'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Math expression"}
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "current_time",
            "description": "Get the current local date and time in ISO format.",
            "parameters": {
                "type": "object",
                "properties": {
                    "timezone": {"type": "string", "description": "Ignored; local time"}
                },
            },
        },
    },
]

_IMPL: dict[str, Callable[..., str]] = {
    "calculator": calculator,
    "current_time": current_time,
}


def dispatch(name: str, arguments: dict[str, Any]) -> str:
    """Execute a tool call requested by the model."""
    if name not in _IMPL:
        return f"Error: unknown tool '{name}'"
    try:
        return _IMPL[name](**arguments)
    except Exception as exc:  # surfaced back to the model as an observation
        return f"Error running {name}: {exc}"
