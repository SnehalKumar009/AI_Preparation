"""Calculator MCP server (notebook 17).

The smallest useful server: one ``evaluate`` tool that computes an arithmetic
expression. It uses a restricted :mod:`ast` evaluator — never :func:`eval` — so a
hostile expression cannot run arbitrary code. This is the reference for how a
server exposes a single, well-typed tool.
"""

from __future__ import annotations

import ast
import operator as op

from mcp_lab.miniserver import MiniMCPServer

# Only these node/operator types are allowed through the evaluator.
_BIN_OPS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.FloorDiv: op.floordiv,
    ast.Mod: op.mod,
    ast.Pow: op.pow,
}
_UNARY_OPS = {ast.UAdd: op.pos, ast.USub: op.neg}


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        return _BIN_OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_node(node.operand))
    raise ValueError("unsupported expression")


def evaluate(expression: str) -> str:
    """Evaluate a basic arithmetic expression, e.g. '2 * (3 + 4)'."""
    tree = ast.parse(expression, mode="eval")
    return str(_eval_node(tree))


def build_server() -> MiniMCPServer:
    server = MiniMCPServer("calculator")
    server.tool(
        name="evaluate",
        description="Evaluate a basic arithmetic expression, e.g. '2 * (3 + 4)'.",
        input_schema={
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "Math expression"}
            },
            "required": ["expression"],
        },
    )(evaluate)
    return server


if __name__ == "__main__":
    build_server().serve_stdio()
