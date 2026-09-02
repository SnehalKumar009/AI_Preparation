"""A minimal MCP client, from scratch (notebook 07).

The client side is the mirror of :class:`~mcp_lab.miniserver.MiniMCPServer`. It
launches a server as a **subprocess**, then talks JSON-RPC 2.0 over that child's
stdin/stdout: send ``initialize``, discover tools with ``tools/list``, and invoke
them with ``tools/call``. This is exactly what an MCP *host* (an editor, a chat
app, an agent) does under the hood — spawn a server and speak the protocol.

Use it as a context manager so the subprocess is always cleaned up::

    with MiniMCPClient([sys.executable, "-m", "mcp_lab.servers.calculator"]) as c:
        print(c.list_tools())
        print(c.call_tool("evaluate", {"expression": "2 + 2"}))
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

from mcp_lab.protocol import (
    INITIALIZE,
    INITIALIZED,
    PROTOCOL_VERSION,
    TOOLS_CALL,
    TOOLS_LIST,
    make_notification,
    make_request,
)
from mcp_lab.transport import StdioTransport

# Phase 4 root, so spawned servers resolve ``python -m mcp_lab.servers.*``.
_PHASE4_ROOT = Path(__file__).resolve().parent.parent


class MiniMCPClient:
    """Spawn one MCP server subprocess and speak JSON-RPC 2.0 to it."""

    def __init__(
        self,
        command: list[str],
        client_name: str = "mcp_lab.miniclient",
        cwd: str | None = None,
    ) -> None:
        self.command = command
        self.client_name = client_name
        self._cwd = cwd or str(_PHASE4_ROOT)
        self._next_id = 0
        self._proc: subprocess.Popen[str] | None = None
        self._transport: StdioTransport | None = None
        self.server_info: dict[str, Any] = {}

    # ---- lifecycle --------------------------------------------------------
    def start(self) -> "MiniMCPClient":
        self._proc = subprocess.Popen(
            self.command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=sys.stderr,  # server logs pass through, never mixed into stdout
            text=True,
            bufsize=1,
            cwd=self._cwd,
        )
        assert self._proc.stdin and self._proc.stdout
        self._transport = StdioTransport(reader=self._proc.stdout, writer=self._proc.stdin)
        self._initialize()
        return self

    def close(self) -> None:
        if self._proc is None:
            return
        try:
            if self._proc.stdin:
                self._proc.stdin.close()
            self._proc.wait(timeout=5)
        except Exception:
            self._proc.kill()
        finally:
            self._proc = None
            self._transport = None

    def __enter__(self) -> "MiniMCPClient":
        return self.start()

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ---- protocol ---------------------------------------------------------
    def _request(self, method: str, params: dict[str, Any] | None = None) -> Any:
        assert self._transport is not None, "client not started"
        self._next_id += 1
        req_id = self._next_id
        self._transport.send(make_request(req_id, method, params))
        # Skip any notifications; return the response matching our id.
        while True:
            msg = self._transport.receive()
            if msg is None:
                raise ConnectionError("server closed the connection")
            if msg.get("id") != req_id:
                continue
            if "error" in msg:
                err = msg["error"]
                raise RuntimeError(f"MCP error {err.get('code')}: {err.get('message')}")
            return msg.get("result")

    def _notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        assert self._transport is not None, "client not started"
        self._transport.send(make_notification(method, params))

    def _initialize(self) -> None:
        self.server_info = self._request(
            INITIALIZE,
            {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": self.client_name, "version": "0.1.0"},
            },
        )
        self._notify(INITIALIZED)

    # ---- high-level API ---------------------------------------------------
    def list_tools(self) -> list[dict[str, Any]]:
        return self._request(TOOLS_LIST).get("tools", [])

    def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> str:
        """Call a tool and return its text content (raises on ``isError``)."""
        result = self._request(TOOLS_CALL, {"name": name, "arguments": arguments or {}})
        text = "\n".join(
            block.get("text", "")
            for block in result.get("content", [])
            if block.get("type") == "text"
        )
        if result.get("isError"):
            raise RuntimeError(text or f"tool '{name}' failed")
        return text
