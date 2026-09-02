"""Servers package: each module is a standalone MCP server (notebooks 13–17, 22).

Run any server over stdio with, e.g.::

    python -m mcp_lab.servers.calculator

They are built on :class:`mcp_lab.miniserver.MiniMCPServer`, so a
:class:`mcp_lab.miniclient.MiniMCPClient` (or the multi-server
:class:`mcp_lab.client.MCPClient`) can spawn and drive them.
"""
