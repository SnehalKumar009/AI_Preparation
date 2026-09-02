"""Filesystem MCP server (notebook 13).

Exposes read-only filesystem access **confined to a sandbox root**. Every path
argument is resolved through :func:`mcp_lab.safety.resolve_within_root`, so a
client cannot read outside the root with ``../`` or an absolute path. The root
comes from ``MCP_FS_ROOT`` (default: this phase's folder).

Tools: ``read_file``, ``list_dir``, ``search_files``.
"""

from __future__ import annotations

import os
from pathlib import Path

from mcp_lab.miniserver import MiniMCPServer
from mcp_lab.safety import resolve_within_root

_ROOT = Path(os.getenv("MCP_FS_ROOT", Path(__file__).resolve().parent.parent.parent)).resolve()
_MAX_BYTES = 100_000


def read_file(path: str) -> str:
    """Return the text contents of a file under the sandbox root."""
    target = resolve_within_root(_ROOT, path)
    if not target.is_file():
        return f"Error: not a file: {path}"
    data = target.read_text(encoding="utf-8", errors="replace")
    if len(data) > _MAX_BYTES:
        return data[:_MAX_BYTES] + f"\n... [truncated at {_MAX_BYTES} bytes]"
    return data


def list_dir(path: str = ".") -> str:
    """List entries of a directory under the sandbox root (dirs end with '/')."""
    target = resolve_within_root(_ROOT, path)
    if not target.is_dir():
        return f"Error: not a directory: {path}"
    names = sorted(
        f"{p.name}/" if p.is_dir() else p.name for p in target.iterdir()
    )
    return "\n".join(names) if names else "(empty)"


def search_files(pattern: str, path: str = ".") -> str:
    """Find files whose name matches a glob pattern, recursively, under a dir."""
    target = resolve_within_root(_ROOT, path)
    if not target.is_dir():
        return f"Error: not a directory: {path}"
    matches = [str(p.relative_to(_ROOT)) for p in target.rglob(pattern) if p.is_file()]
    return "\n".join(sorted(matches)[:200]) if matches else "(no matches)"


def build_server() -> MiniMCPServer:
    server = MiniMCPServer("filesystem")
    server.tool(
        name="read_file",
        description="Read a UTF-8 text file under the sandbox root.",
        input_schema={
            "type": "object",
            "properties": {"path": {"type": "string", "description": "Path under root"}},
            "required": ["path"],
        },
    )(read_file)
    server.tool(
        name="list_dir",
        description="List entries of a directory under the sandbox root.",
        input_schema={
            "type": "object",
            "properties": {"path": {"type": "string", "description": "Directory path"}},
        },
    )(list_dir)
    server.tool(
        name="search_files",
        description="Recursively find files matching a glob pattern (e.g. '*.py').",
        input_schema={
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Glob, e.g. '*.py'"},
                "path": {"type": "string", "description": "Directory to search"},
            },
            "required": ["pattern"],
        },
    )(search_files)
    return server


if __name__ == "__main__":
    build_server().serve_stdio()
