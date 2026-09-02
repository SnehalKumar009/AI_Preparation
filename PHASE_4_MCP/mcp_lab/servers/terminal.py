"""Terminal MCP server (notebook 16).

Runs shell commands, but only from an **allowlist** and always with a timeout,
and never through a real shell (``shell=False`` + :func:`shlex.split`) so shell
metacharacters cannot chain extra commands. The allowlist comes from
``MCP_TERMINAL_ALLOW`` (comma-separated) or a safe read-only default. Working
directory is ``MCP_TERMINAL_CWD`` (default: this phase's folder).

Tool: ``run_command``.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from mcp_lab.miniserver import MiniMCPServer
from mcp_lab.safety import is_allowed_command, split_command

_DEFAULT_ALLOW = {"ls", "cat", "echo", "pwd", "git", "grep", "find", "wc", "head", "tail"}
_ALLOW = (
    {c.strip() for c in os.environ["MCP_TERMINAL_ALLOW"].split(",") if c.strip()}
    if os.getenv("MCP_TERMINAL_ALLOW")
    else _DEFAULT_ALLOW
)
_CWD = Path(os.getenv("MCP_TERMINAL_CWD", Path(__file__).resolve().parent.parent.parent)).resolve()
_TIMEOUT = 10


def run_command(command: str) -> str:
    """Run an allowlisted shell command with a timeout and return its output."""
    if not is_allowed_command(command, _ALLOW):
        allowed = ", ".join(sorted(_ALLOW))
        return f"Error: command not allowed. Allowed programs: {allowed}"
    try:
        proc = subprocess.run(
            split_command(command),
            cwd=_CWD,
            capture_output=True,
            text=True,
            timeout=_TIMEOUT,
            shell=False,
        )
    except subprocess.TimeoutExpired:
        return f"Error: command timed out after {_TIMEOUT}s"
    except FileNotFoundError:
        return "Error: program not found"
    out = (proc.stdout or "") + (proc.stderr or "")
    return out.strip() or f"(exit {proc.returncode}, no output)"


def build_server() -> MiniMCPServer:
    server = MiniMCPServer("terminal")
    server.tool(
        name="run_command",
        description="Run an allowlisted, read-only shell command with a timeout.",
        input_schema={
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "e.g. 'ls -la' or 'grep -r TODO .'"}
            },
            "required": ["command"],
        },
    )(run_command)
    return server


if __name__ == "__main__":
    build_server().serve_stdio()
