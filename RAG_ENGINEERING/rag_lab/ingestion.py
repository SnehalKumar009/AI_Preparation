"""Document ingestion & parsing (notebook 18).

The first stage of any RAG pipeline: turn messy source files (markdown, HTML,
plain text, PDF) into clean, normalized text that is safe to chunk and embed.
Parsers for heavy formats (PDF, HTML) import their dependency lazily, so the
module loads even when those libraries are absent — you only pay for what you use.
"""

from __future__ import annotations

import re
from pathlib import Path

from rag_lab.chunking import Chunk, recursive_chunks

# Collapse runs of blank lines / whitespace introduced by format converters.
_MULTISPACE = re.compile(r"[ \t]+")
_MULTINEWLINE = re.compile(r"\n{3,}")
_HTML_TAG = re.compile(r"<[^>]+>")
_MD_ARTIFACT = re.compile(r"[#>*_`]{1,}")


def clean_text(text: str) -> str:
    """Normalize whitespace so downstream chunking sees consistent input."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _MULTISPACE.sub(" ", text)
    text = _MULTINEWLINE.sub("\n\n", text)
    return text.strip()


def parse_text(raw: str) -> str:
    """Plain text just needs whitespace normalization."""
    return clean_text(raw)


def parse_markdown(raw: str) -> str:
    """Strip common markdown markup down to readable prose."""
    return clean_text(_MD_ARTIFACT.sub("", raw))


def parse_html(raw: str) -> str:
    """Extract visible text from HTML. Uses BeautifulSoup if present, else regex."""
    try:
        from bs4 import BeautifulSoup  # lazy: only needed for .html sources
    except ImportError:
        return clean_text(_HTML_TAG.sub(" ", raw))
    return clean_text(BeautifulSoup(raw, "html.parser").get_text(" "))


def parse_pdf(path: str | Path) -> str:
    """Extract text from a PDF (requires ``pypdf``)."""
    from pypdf import PdfReader  # lazy: only needed for .pdf sources

    reader = PdfReader(str(path))
    return clean_text("\n\n".join(page.extract_text() or "" for page in reader.pages))


#: Map file extension -> a parser that takes raw text.
_TEXT_PARSERS = {
    ".md": parse_markdown,
    ".markdown": parse_markdown,
    ".txt": parse_text,
    ".html": parse_html,
    ".htm": parse_html,
}


def parse_document(path: str | Path) -> str:
    """Dispatch to the right parser by file extension and return clean text."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return parse_pdf(path)
    parser = _TEXT_PARSERS.get(suffix, parse_text)
    return parser(path.read_text(encoding="utf-8", errors="ignore"))


def ingest_path(
    path: str | Path,
    chunk_size: int = 200,
    overlap: int = 40,
) -> list[Chunk]:
    """Parse one file and split it into chunks tagged with the source filename."""
    path = Path(path)
    text = parse_document(path)
    return recursive_chunks(text, source=path.name, chunk_size=chunk_size, overlap=overlap)


def ingest_directory(
    directory: str | Path,
    patterns: tuple[str, ...] = ("*.md", "*.txt", "*.html", "*.pdf"),
    chunk_size: int = 200,
    overlap: int = 40,
) -> list[Chunk]:
    """Ingest every matching file in a directory into one flat list of chunks."""
    directory = Path(directory)
    chunks: list[Chunk] = []
    for pattern in patterns:
        for path in sorted(directory.glob(pattern)):
            chunks.extend(ingest_path(path, chunk_size=chunk_size, overlap=overlap))
    return chunks
