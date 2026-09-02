"""A minimal MCP server, from scratch (notebook 06).

This is the whole server side of MCP in one small class: register tools (each a
Python function plus the JSON schema the model sees), answer the ``initialize``
handshake, list tools, and run a ``tools/call``. It speaks JSON-RPC 2.0 over
stdio via :mod:`mcp_lab.transport`. The five real servers in
:mod:`mcp_lab.servers` are just this class with domain tools attached.

Design choices kept deliberately small:

- Tool results are returned in MCP's ``content`` shape (a list of
  ``{"type": "text", "text": ...}`` blocks) so a real MCP client can read them.
- A handler that raises :class:`~mcp_lab.protocol.JsonRpcError` becomes a proper
  JSON-RPC error; any other exception becomes an ``isError`` tool result so one
  broken tool never kills the connection.
- ``resources`` and ``prompts`` are optional and default to empty lists.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Any, Callable

from mcp_lab.protocol import (
    INITIALIZE,
    INTERNAL_ERROR,
    INVALID_PARAMS,
    METHOD_NOT_FOUND,
    PROMPTS_LIST,
    PROTOCOL_VERSION,
    RESOURCES_LIST,
    TOOLS_CALL,
    TOOLS_LIST,
    JsonRpcError,
    make_error,
    make_response,
)
from mcp_lab.transport import StdioTransport, read_message, write_message


@dataclass
class ToolSpec:
    """A server-side tool: a callable plus the schema advertised to clients."""

    name: str
    description: str
    func: Callable[..., Any]
    input_schema: dict[str, Any] = field(
        default_factory=lambda: {"type": "object", "properties": {}}
    )
    # Behavior hints (readOnlyHint / destructiveHint / idempotentHint) — see nb 34.
    annotations: dict[str, Any] = field(default_factory=dict)

    def to_wire(self) -> dict[str, Any]:
        wire: dict[str, Any] = {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_schema,
        }
        if self.annotations:
            wire["annotations"] = self.annotations
        return wire


class MiniMCPServer:
    """A tiny MCP server that talks JSON-RPC 2.0 over stdio."""

    def __init__(self, name: str, version: str = "0.1.0") -> None:
        self.name = name
        self.version = version
        self._tools: dict[str, ToolSpec] = {}

    # ---- registration -----------------------------------------------------
    def tool(
        self,
        name: str | None = None,
        description: str | None = None,
        input_schema: dict[str, Any] | None = None,
        annotations: dict[str, Any] | None = None,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Decorator: register the wrapped function as an MCP tool."""

        def wrap(fn: Callable[..., Any]) -> Callable[..., Any]:
            self.add_tool(
                ToolSpec(
                    name=name or fn.__name__,
                    description=description or (fn.__doc__ or "").strip().split("\n")[0],
                    func=fn,
                    input_schema=input_schema or {"type": "object", "properties": {}},
                    annotations=annotations or {},
                )
            )
            return fn

        return wrap

    def add_tool(self, spec: ToolSpec) -> ToolSpec:
        self._tools[spec.name] = spec
        return spec

    # ---- request handling -------------------------------------------------
    def handle(self, message: dict[str, Any]) -> dict[str, Any] | None:
        """Route one request to a result. Returns ``None`` for notifications."""
        method = message.get("method")
        msg_id = message.get("id")

        # Notifications (no id) are acknowledged by doing nothing.
        if msg_id is None:
            return None

        try:
            result = self._dispatch(method, message.get("params") or {})
            return make_response(msg_id, result)
        except JsonRpcError as exc:
            return make_error(msg_id, exc.code, exc.message, exc.data)
        except Exception as exc:  # never let one bad call crash the loop
            return make_error(msg_id, INTERNAL_ERROR, f"{type(exc).__name__}: {exc}")

    def _dispatch(self, method: str | None, params: dict[str, Any]) -> Any:
        if method == INITIALIZE:
            return {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"tools": {}, "resources": {}, "prompts": {}},
                "serverInfo": {"name": self.name, "version": self.version},
            }
        if method == TOOLS_LIST:
            return {"tools": [t.to_wire() for t in self._tools.values()]}
        if method == TOOLS_CALL:
            return self._call_tool(params)
        if method == RESOURCES_LIST:
            return {"resources": []}
        if method == PROMPTS_LIST:
            return {"prompts": []}
        raise JsonRpcError(METHOD_NOT_FOUND, f"Unknown method: {method}")

    def _call_tool(self, params: dict[str, Any]) -> dict[str, Any]:
        name = params.get("name")
        args = params.get("arguments") or {}
        tool = self._tools.get(name)
        if tool is None:
            raise JsonRpcError(INVALID_PARAMS, f"Unknown tool: {name}")
        try:
            output = tool.func(**args)
            return {
                "content": [{"type": "text", "text": str(output)}],
                "isError": False,
            }
        except Exception as exc:
            # Tool failures are reported in-band so the model can recover.
            return {
                "content": [{"type": "text", "text": f"Error: {exc}"}],
                "isError": True,
            }

    # ---- run loop ---------------------------------------------------------
    def serve(self, transport: StdioTransport) -> None:
        """Read requests until the input stream closes."""
        while True:
            message = transport.receive()
            if message is None:
                break
            if not message:
                continue
            reply = self.handle(message)
            if reply is not None:
                transport.send(reply)

    def serve_stdio(self) -> None:
        """Serve over this process's stdin/stdout (the ``__main__`` entry point)."""
        self.serve(StdioTransport(reader=sys.stdin, writer=sys.stdout))
