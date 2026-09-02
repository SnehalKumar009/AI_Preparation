"""SQLite MCP server (notebook 15).

Read-only access to a SQLite database. The DB path comes from ``MCP_SQLITE_DB``;
if unset, a small demo database is created in a temp file and seeded so the
server is useful out of the box. Two defenses keep this read-only:

- the connection is opened in ``mode=ro`` (URI), and
- ``run_select`` rejects anything that is not a single ``SELECT`` statement.

Tools: ``list_tables``, ``describe_table``, ``run_select``.
"""

from __future__ import annotations

import os
import sqlite3
import tempfile
from pathlib import Path

from mcp_lab.miniserver import MiniMCPServer


def _demo_db_path() -> str:
    """Create and seed a throwaway demo DB once; reuse it across calls."""
    path = Path(tempfile.gettempdir()) / "mcp_lab_demo.sqlite"
    if not path.exists():
        con = sqlite3.connect(path)
        con.executescript(
            """
            CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, email TEXT);
            CREATE TABLE orders (id INTEGER PRIMARY KEY, user_id INTEGER, total REAL);
            INSERT INTO users (name, email) VALUES
                ('Ada', 'ada@example.com'), ('Alan', 'alan@example.com');
            INSERT INTO orders (user_id, total) VALUES (1, 42.0), (1, 8.5), (2, 99.0);
            """
        )
        con.commit()
        con.close()
    return str(path)


_DB_PATH = os.getenv("MCP_SQLITE_DB") or _demo_db_path()


def _connect() -> sqlite3.Connection:
    # Read-only URI connection: writes fail at the driver level.
    return sqlite3.connect(f"file:{_DB_PATH}?mode=ro", uri=True)


def list_tables() -> str:
    """List the names of all tables in the database."""
    con = _connect()
    try:
        rows = con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
    finally:
        con.close()
    return "\n".join(r[0] for r in rows) if rows else "(no tables)"


def describe_table(table: str) -> str:
    """Show column names and types for a table."""
    con = _connect()
    try:
        rows = con.execute(f"PRAGMA table_info({table})").fetchall()
    finally:
        con.close()
    if not rows:
        return f"(no such table: {table})"
    return "\n".join(f"{r[1]} {r[2]}" for r in rows)


def run_select(query: str, limit: int = 50) -> str:
    """Run a single read-only SELECT and return rows (capped by ``limit``)."""
    stripped = query.strip().rstrip(";")
    if ";" in stripped:
        return "Error: only a single statement is allowed"
    if not stripped.lower().startswith("select"):
        return "Error: only SELECT queries are allowed"
    con = _connect()
    try:
        cur = con.execute(stripped)
        cols = [d[0] for d in cur.description]
        rows = cur.fetchmany(int(limit))
    except sqlite3.Error as exc:
        return f"Error: {exc}"
    finally:
        con.close()
    header = " | ".join(cols)
    body = "\n".join(" | ".join(str(v) for v in row) for row in rows)
    return f"{header}\n{body}" if rows else f"{header}\n(no rows)"


def build_server() -> MiniMCPServer:
    server = MiniMCPServer("sqlite")
    server.tool(
        name="list_tables",
        description="List all table names in the database.",
        input_schema={"type": "object", "properties": {}},
    )(list_tables)
    server.tool(
        name="describe_table",
        description="Show columns and types for a table.",
        input_schema={
            "type": "object",
            "properties": {"table": {"type": "string", "description": "Table name"}},
            "required": ["table"],
        },
    )(describe_table)
    server.tool(
        name="run_select",
        description="Run a single read-only SELECT query.",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "A SELECT statement"},
                "limit": {"type": "integer", "description": "Max rows (default 50)"},
            },
            "required": ["query"],
        },
    )(run_select)
    return server


if __name__ == "__main__":
    build_server().serve_stdio()
