"""Git MCP server (notebook 14).

Read-only introspection of a git repository via GitPython. The repo root comes
from ``MCP_GIT_ROOT`` (default: the workspace containing this phase). All tools
are read-only — no commit, checkout, or push — so an agent can *understand*
history without changing it.

Tools: ``git_status``, ``git_log``, ``git_diff``, ``git_blame``.
GitPython is imported lazily so the module loads even if it is not installed.
"""

from __future__ import annotations

import os
from pathlib import Path

from mcp_lab.miniserver import MiniMCPServer

_ROOT = Path(os.getenv("MCP_GIT_ROOT", Path(__file__).resolve().parent.parent.parent)).resolve()


def _repo():
    from git import Repo  # lazy import: only needed when a tool runs

    return Repo(_ROOT, search_parent_directories=True)


def git_status() -> str:
    """Show changed, staged, and untracked files."""
    repo = _repo()
    lines = []
    if repo.is_dirty(untracked_files=True):
        for item in repo.index.diff(None):
            lines.append(f"modified: {item.a_path}")
        for item in repo.index.diff("HEAD"):
            lines.append(f"staged:   {item.a_path}")
        for path in repo.untracked_files:
            lines.append(f"untracked:{path}")
    else:
        lines.append("clean")
    return "\n".join(lines)


def git_log(max_count: int = 10) -> str:
    """Show the most recent commits (hash, author, subject)."""
    repo = _repo()
    out = []
    for c in repo.iter_commits(max_count=int(max_count)):
        out.append(f"{c.hexsha[:8]} {c.author.name}: {c.summary}")
    return "\n".join(out) if out else "(no commits)"


def git_diff(path: str | None = None) -> str:
    """Show the working-tree diff, optionally limited to one path."""
    repo = _repo()
    diff = repo.git.diff(path) if path else repo.git.diff()
    return diff or "(no changes)"


def git_blame(path: str) -> str:
    """Show line-by-line last-commit attribution for a file."""
    repo = _repo()
    out = []
    for commit, lines in repo.blame("HEAD", path):
        for line in lines:
            out.append(f"{commit.hexsha[:8]} {commit.author.name}: {line}")
    return "\n".join(out[:200]) if out else "(empty)"


def build_server() -> MiniMCPServer:
    server = MiniMCPServer("git")
    server.tool(
        name="git_status",
        description="Show changed, staged, and untracked files.",
        input_schema={"type": "object", "properties": {}},
    )(git_status)
    server.tool(
        name="git_log",
        description="Show the most recent commits.",
        input_schema={
            "type": "object",
            "properties": {"max_count": {"type": "integer", "description": "How many commits"}},
        },
    )(git_log)
    server.tool(
        name="git_diff",
        description="Show the working-tree diff, optionally for one path.",
        input_schema={
            "type": "object",
            "properties": {"path": {"type": "string", "description": "Optional path filter"}},
        },
    )(git_diff)
    server.tool(
        name="git_blame",
        description="Show last-commit attribution per line for a file.",
        input_schema={
            "type": "object",
            "properties": {"path": {"type": "string", "description": "File to blame"}},
            "required": ["path"],
        },
    )(git_blame)
    return server


if __name__ == "__main__":
    build_server().serve_stdio()
