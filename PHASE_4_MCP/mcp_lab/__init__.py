"""mcp_lab: the Phase 4 Model Context Protocol core.

Reuses Phase 1 ``study_buddy`` (providers + cost tracking), Phase 2 ``rag_lab``
(retrieval + guardrails for the code-search server), and Phase 3 ``agents_lab``
(the agent loop + tool registry the MCP tools are bridged into). Importing
``mcp_lab`` locates the sibling folders that hold those packages and puts them on
``sys.path`` automatically (by content, not folder name), so notebooks only need
to add this package's own root to the path.

The package is built **from scratch first** so the wire mechanics stay visible:
``protocol`` (JSON-RPC 2.0), ``transport`` (stdio framing), ``miniserver`` /
``miniclient`` (a minimal MCP server and client). The five servers in
``mcp_lab.servers`` run on that core. The official ``mcp`` SDK is introduced in
the notebooks (08–12); ``client`` and ``agent_bridge`` are imported on demand so
``import mcp_lab`` stays light and never forces the heavy cloud SDKs.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_PKG_DIR = _Path(__file__).resolve().parent      # .../<phase4>/mcp_lab
_PHASE4_ROOT = _PKG_DIR.parent                    # .../<phase4>

# Sibling packages this phase reuses, matched by content (folder name-agnostic).
_TARGETS = ("study_buddy", "rag_lab", "agents_lab")


def _bootstrap_siblings() -> None:
    """Add sibling folders holding the reused packages to sys.path."""
    found: set[str] = set()
    for base in [_PHASE4_ROOT.parent, *_PHASE4_ROOT.parents]:
        try:
            siblings = [s for s in base.iterdir() if s.is_dir()]
        except (PermissionError, OSError):
            continue
        for sibling in siblings:
            for target in _TARGETS:
                if target not in found and (sibling / target).exists():
                    if str(sibling) not in _sys.path:
                        _sys.path.insert(0, str(sibling))
                    found.add(target)
        if len(found) == len(_TARGETS):
            return
    missing = [t for t in _TARGETS if t not in found]
    if "study_buddy" in missing:
        raise ImportError(
            "Could not locate a sibling folder containing study_buddy. "
            "Phase 4 reuses Phases 1–3; keep them as sibling folders."
        )


_bootstrap_siblings()

from mcp_lab.protocol import (
    PROTOCOL_VERSION,
    JsonRpcError,
    make_error,
    make_notification,
    make_request,
    make_response,
)
from mcp_lab.transport import StdioTransport, read_message, write_message
from mcp_lab.miniserver import MiniMCPServer, ToolSpec
from mcp_lab.miniclient import MiniMCPClient
from mcp_lab.client import MCPClient
from mcp_lab import safety

__all__ = [
    "PROTOCOL_VERSION",
    "JsonRpcError",
    "make_request",
    "make_response",
    "make_error",
    "make_notification",
    "StdioTransport",
    "read_message",
    "write_message",
    "MiniMCPServer",
    "ToolSpec",
    "MiniMCPClient",
    "MCPClient",
    "safety",
]
