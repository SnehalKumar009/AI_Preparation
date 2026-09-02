"""stdio transport: newline-delimited JSON-RPC framing (notebooks 03 & 06).

MCP's stdio transport is deliberately simple: each JSON-RPC message is written as
**one line of JSON** to a stream and read back the same way. A server reads
requests from ``stdin`` and writes responses to ``stdout``; anything it prints to
``stderr`` is free-form logging that never touches the protocol. Keeping the
framing this small is what lets the minimal server and client in this package fit
in a single notebook cell each.
"""

from __future__ import annotations

import json
from typing import Any, IO


def write_message(stream: IO[str], message: dict[str, Any]) -> None:
    """Serialize one message as a single line and flush so it is sent at once."""
    stream.write(json.dumps(message, separators=(",", ":")) + "\n")
    stream.flush()


def read_message(stream: IO[str]) -> dict[str, Any] | None:
    """Read one line and parse it as JSON. ``None`` means the stream closed."""
    line = stream.readline()
    if not line:
        return None
    line = line.strip()
    if not line:
        return {}
    return json.loads(line)


class StdioTransport:
    """A reader/writer pair over two text streams (a server's or subprocess's)."""

    def __init__(self, reader: IO[str], writer: IO[str]) -> None:
        self.reader = reader
        self.writer = writer

    def send(self, message: dict[str, Any]) -> None:
        write_message(self.writer, message)

    def receive(self) -> dict[str, Any] | None:
        return read_message(self.reader)
