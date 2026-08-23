"""Load the markdown corpus into metadata-rich chunks.

Each file may start with a simple ``--- ... ---`` YAML-ish front-matter block
(title, tags, audience). We split the body on ``##`` headings so every chunk
carries its source file, section heading, and the file's tags — exactly the
metadata later notebooks filter on.
"""

from __future__ import annotations

import re
from pathlib import Path

from rag_lab import NOTES_DIR
from rag_lab.chunking import Chunk

_FRONT = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_HEADING = re.compile(r"^##\s+(.*)$", re.MULTILINE)


def _parse_front_matter(text: str) -> tuple[dict[str, str | list[str]], str]:
    m = _FRONT.match(text)
    if not m:
        return {}, text
    meta: dict[str, str | list[str]] = {}
    for line in m.group(1).splitlines():
        if ":" not in line:
            continue
        key, val = line.split(":", 1)
        val = val.strip()
        if val.startswith("[") and val.endswith("]"):
            meta[key.strip()] = [v.strip() for v in val[1:-1].split(",") if v.strip()]
        else:
            meta[key.strip()] = val
    return meta, text[m.end():]


def load_markdown(directory: str | Path | None = None) -> list[Chunk]:
    """Return one Chunk per ``##`` section across all .md files in a directory."""
    directory = Path(directory) if directory else NOTES_DIR
    chunks: list[Chunk] = []
    for path in sorted(directory.glob("*.md")):
        meta, body = _parse_front_matter(path.read_text(encoding="utf-8"))
        headings = list(_HEADING.finditer(body))
        for i, h in enumerate(headings):
            start = h.end()
            end = headings[i + 1].start() if i + 1 < len(headings) else len(body)
            section = body[start:end].strip()
            if not section:
                continue
            chunks.append(
                Chunk(
                    text=section,
                    source=path.name,
                    index=i,
                    meta={"heading": h.group(1).strip(), **meta},
                )
            )
    return chunks
