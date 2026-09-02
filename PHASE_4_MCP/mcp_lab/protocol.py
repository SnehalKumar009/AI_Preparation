"""JSON-RPC 2.0 message helpers + MCP method/constant names (notebook 04).

MCP speaks **JSON-RPC 2.0**: every message is a small JSON object. There are
three shapes — a *request* (has an ``id``, expects a reply), a *notification*
(no ``id``, fire-and-forget), and a *response* (carries ``result`` or ``error``
for a matching ``id``). This module builds those shapes and nothing else, so the
wire format stays obvious in the notebooks.
"""

from __future__ import annotations

from typing import Any

#: MCP protocol revision this lab implements.
PROTOCOL_VERSION = "2024-11-05"

# ---- Standard JSON-RPC 2.0 error codes ------------------------------------
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603

# ---- MCP method names (the subset this lab implements) --------------------
INITIALIZE = "initialize"
INITIALIZED = "notifications/initialized"
TOOLS_LIST = "tools/list"
TOOLS_CALL = "tools/call"
RESOURCES_LIST = "resources/list"
RESOURCES_READ = "resources/read"
PROMPTS_LIST = "prompts/list"
PROMPTS_GET = "prompts/get"
PING = "ping"


class JsonRpcError(Exception):
    """A JSON-RPC error raised inside a handler; carries a wire error code."""

    def __init__(self, code: int, message: str, data: Any = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.data = data

    def to_dict(self) -> dict[str, Any]:
        err: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.data is not None:
            err["data"] = self.data
        return err


def make_request(id: int | str, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    """A request expects a response with the same ``id``."""
    msg: dict[str, Any] = {"jsonrpc": "2.0", "id": id, "method": method}
    if params is not None:
        msg["params"] = params
    return msg


def make_notification(method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    """A notification has no ``id`` and gets no response."""
    msg: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
    if params is not None:
        msg["params"] = params
    return msg


def make_response(id: int | str, result: Any) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": id, "result": result}


def make_error(id: int | str | None, code: int, message: str, data: Any = None) -> dict[str, Any]:
    err: dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        err["data"] = data
    return {"jsonrpc": "2.0", "id": id, "error": err}


def is_request(msg: dict[str, Any]) -> bool:
    return "method" in msg and "id" in msg


def is_notification(msg: dict[str, Any]) -> bool:
    return "method" in msg and "id" not in msg
