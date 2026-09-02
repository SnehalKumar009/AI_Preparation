"""Advanced MCP mechanics: bidirectional messaging (notebooks 28–34).

The subprocess mini server/client (notebooks 06–07) cover the request→response
core. The advanced spec features — **sampling**, **roots**, **progress**,
**logging**, **list-changed**, **cancellation** — all need *bidirectional*
traffic: the server sends **requests** back to the client (sampling, roots) and
**notifications** to it (progress, logging, list-changed).

To keep those flows faithful *and* runnable in a single cell, this module wires a
server and a client together **in process** via :class:`Session`. A tool receives
a :class:`ServerContext` (``ctx``) it can use to talk back to the client:

    session = Session(sampling_handler=my_llm, roots=[...])

    @session.tool(description="Summarize via the host's LLM.")
    def summarize(ctx, text):
        ctx.log("info", "starting summary")
        return ctx.sample(f"Summarize: {text}")   # server -> client -> LLM

    session.call_tool("summarize", {"text": "..."})
    session.notifications                          # what the client received

The message shapes match the real protocol (built with :mod:`mcp_lab.protocol`),
so moving to the official SDK later is a change of transport, not of concepts.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, Callable

from mcp_lab.protocol import make_notification, make_request

# A client-side handler for a server-initiated sampling request.
SamplingHandler = Callable[[dict[str, Any]], dict[str, Any]]


@dataclass
class ServerContext:
    """Handed to an advanced tool so it can talk back to the client."""

    session: "Session"
    progress_token: Any = None

    def progress(self, progress: float, total: float | None = None) -> None:
        """Emit a ``notifications/progress`` update to the client."""
        params: dict[str, Any] = {"progressToken": self.progress_token, "progress": progress}
        if total is not None:
            params["total"] = total
        self.session._deliver_notification(make_notification("notifications/progress", params))

    def log(self, level: str, message: str) -> None:
        """Emit a ``notifications/message`` log record (level: info/warning/error)."""
        self.session._deliver_notification(
            make_notification("notifications/message", {"level": level, "data": message})
        )

    def sample(self, prompt: str, system: str | None = None) -> str:
        """Ask the client's LLM to complete a prompt (``sampling/createMessage``)."""
        params: dict[str, Any] = {
            "messages": [{"role": "user", "content": {"type": "text", "text": prompt}}],
            "maxTokens": 512,
        }
        if system:
            params["systemPrompt"] = system
        result = self.session._request_client("sampling/createMessage", params)
        return result.get("content", {}).get("text", "")

    def list_roots(self) -> list[dict[str, Any]]:
        """Ask the client which filesystem roots it exposes (``roots/list``)."""
        return self.session._request_client("roots/list", {}).get("roots", [])

    @property
    def cancelled(self) -> bool:
        """True if the client has cancelled this request's ``progress_token``."""
        return self.progress_token in self.session.cancelled_tokens


class Cancelled(Exception):
    """Raised inside a tool when its request has been cancelled."""


@dataclass
class Session:
    """An in-process MCP session supporting bidirectional messaging."""

    name: str = "advanced"
    sampling_handler: SamplingHandler | None = None
    roots: list[dict[str, Any]] = field(default_factory=list)
    cancelled_tokens: set[Any] = field(default_factory=set)
    notifications: list[dict[str, Any]] = field(default_factory=list)
    _tools: dict[str, dict[str, Any]] = field(default_factory=dict)
    _ids: Any = field(default_factory=lambda: itertools.count(1))

    # ---- registration -----------------------------------------------------
    def tool(
        self,
        name: str | None = None,
        description: str = "",
        input_schema: dict[str, Any] | None = None,
        annotations: dict[str, Any] | None = None,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def wrap(fn: Callable[..., Any]) -> Callable[..., Any]:
            self._tools[name or fn.__name__] = {
                "func": fn,
                "description": description or (fn.__doc__ or "").strip().split("\n")[0],
                "inputSchema": input_schema or {"type": "object", "properties": {}},
                "annotations": annotations or {},
            }
            return fn

        return wrap

    def list_tools(self) -> list[dict[str, Any]]:
        out = []
        for name, meta in self._tools.items():
            entry = {"name": name, "description": meta["description"], "inputSchema": meta["inputSchema"]}
            if meta["annotations"]:
                entry["annotations"] = meta["annotations"]
            out.append(entry)
        return out

    # ---- client -> server -------------------------------------------------
    def call_tool(self, name: str, arguments: dict[str, Any] | None = None, progress_token: Any = None) -> Any:
        meta = self._tools.get(name)
        if meta is None:
            raise KeyError(f"unknown tool: {name}")
        ctx = ServerContext(self, progress_token=progress_token)
        return meta["func"](ctx=ctx, **(arguments or {}))

    def cancel(self, progress_token: Any) -> None:
        """Client-side cancellation of an in-flight request token."""
        self.cancelled_tokens.add(progress_token)
        self._deliver_notification(
            make_notification("notifications/cancelled", {"requestId": progress_token})
        )

    # ---- server -> client -------------------------------------------------
    def _deliver_notification(self, notification: dict[str, Any]) -> None:
        # The client simply records what the server pushed.
        self.notifications.append(notification)

    def _request_client(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        # The client answers a server-initiated request based on its capabilities.
        req = make_request(next(self._ids), method, params)  # noqa: F841 (shape shown in notebooks)
        if method == "sampling/createMessage":
            if self.sampling_handler is None:
                raise RuntimeError("client declared no sampling capability")
            return self.sampling_handler(params)
        if method == "roots/list":
            return {"roots": self.roots}
        raise RuntimeError(f"client cannot handle server request: {method}")

    def emit_list_changed(self, kind: str = "tools") -> None:
        """Server announces its tool/resource list changed (``*/list_changed``)."""
        self._deliver_notification(make_notification(f"notifications/{kind}/list_changed"))
