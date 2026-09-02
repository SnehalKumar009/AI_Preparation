"""Multi-server MCP client (notebook 18).

A real MCP host rarely talks to a single server — it connects to several at once
(filesystem + git + database + ...) and presents their tools as one flat menu.
:class:`MCPClient` spawns one :class:`~mcp_lab.miniclient.MiniMCPClient` per named
server and **namespaces** each tool as ``server__tool`` so names never collide
and stay valid for provider tool-calling (which allows only ``[A-Za-z0-9_-]``).

    with MCPClient(lab_servers(["calculator", "filesystem"])) as mcp:
        mcp.list_tools()                      # namespaced across both servers
        mcp.call_tool("calculator__evaluate", {"expression": "6 * 7"})
"""

from __future__ import annotations

import sys
from typing import Any

from mcp_lab.miniclient import MiniMCPClient

#: Separator between server name and tool name (provider-tool-name safe).
SEP = "__"

#: The lab's built-in servers, by name -> spawn command.
_ALL_SERVERS = {
    "calculator": [sys.executable, "-m", "mcp_lab.servers.calculator"],
    "filesystem": [sys.executable, "-m", "mcp_lab.servers.filesystem"],
    "git": [sys.executable, "-m", "mcp_lab.servers.git"],
    "sqlite": [sys.executable, "-m", "mcp_lab.servers.sqlite"],
    "terminal": [sys.executable, "-m", "mcp_lab.servers.terminal"],
    "code_search": [sys.executable, "-m", "mcp_lab.servers.code_search"],
}


def lab_servers(names: list[str] | None = None) -> dict[str, list[str]]:
    """Return spawn commands for the named lab servers (default: all)."""
    if names is None:
        return dict(_ALL_SERVERS)
    return {n: _ALL_SERVERS[n] for n in names}


class MCPClient:
    """Manage several MCP servers and expose their tools under one namespace."""

    def __init__(self, servers: dict[str, list[str]] | None = None) -> None:
        self._specs = servers if servers is not None else lab_servers()
        self._clients: dict[str, MiniMCPClient] = {}

    # ---- lifecycle --------------------------------------------------------
    def start(self) -> "MCPClient":
        for name, command in self._specs.items():
            self._clients[name] = MiniMCPClient(command, client_name=f"mcp_lab.{name}").start()
        return self

    def close(self) -> None:
        for client in self._clients.values():
            client.close()
        self._clients.clear()

    def __enter__(self) -> "MCPClient":
        return self.start()

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ---- aggregated tool surface -----------------------------------------
    def list_tools(self) -> list[dict[str, Any]]:
        """Every server's tools, each with a namespaced ``name`` (``server__tool``)."""
        tools: list[dict[str, Any]] = []
        for server, client in self._clients.items():
            for tool in client.list_tools():
                qualified = dict(tool)
                qualified["name"] = f"{server}{SEP}{tool['name']}"
                qualified["server"] = server
                tools.append(qualified)
        return tools

    def call_tool(self, qualified_name: str, arguments: dict[str, Any] | None = None) -> str:
        """Route ``server__tool`` to the right server and invoke it."""
        server, _, tool = qualified_name.partition(SEP)
        client = self._clients.get(server)
        if client is None:
            return f"Error: unknown server '{server}'. Known: {list(self._clients)}"
        return client.call_tool(tool, arguments or {})

    @property
    def servers(self) -> list[str]:
        return list(self._clients)
