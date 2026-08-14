"""Multimodal RAG (notebook 29).

Text embeddings can't search images directly. The pragmatic pattern: use a vision
model to **caption** each image, then index the captions as text and retrieve
normally — a query matches the caption, and you return the image it describes.
:func:`caption_image` calls a vision-capable provider (Gemini or Anthropic) via its
raw SDK; :class:`MultimodalStore` indexes captions over the usual NumPy store so
image retrieval reuses everything from earlier notebooks.
"""

from __future__ import annotations

import base64
import mimetypes
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from study_buddy import get_provider

from rag_lab.chunking import Chunk
from rag_lab.vectorstore import Hit, NumpyStore

_CAPTION = "Describe this image in one or two factual sentences for search indexing."


@dataclass
class ImageDoc:
    id: str
    caption: str
    path: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)


def caption_image(
    path: str | Path,
    provider: str = "gemini",
    model: str | None = None,
    prompt: str = _CAPTION,
) -> str:
    """Caption an image with a vision model (Gemini or Anthropic). Needs a key."""
    path = Path(path)
    data = path.read_bytes()
    mime = mimetypes.guess_type(str(path))[0] or "image/png"
    llm = get_provider(provider)

    if provider == "gemini":
        from google.genai import types

        resp = llm.client.models.generate_content(
            model=model or llm.default_model,
            contents=[prompt, types.Part.from_bytes(data=data, mime_type=mime)],
        )
        return (resp.text or "").strip()

    if provider == "anthropic":
        b64 = base64.standard_b64encode(data).decode()
        resp = llm.client.messages.create(
            model=model or llm.default_model,
            max_tokens=256,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image", "source": {"type": "base64", "media_type": mime, "data": b64}},
                ],
            }],
        )
        return "".join(b.text for b in resp.content if b.type == "text").strip()

    raise ValueError("caption_image supports provider 'gemini' or 'anthropic'.")


class MultimodalStore:
    """Retrieve images by embedding and searching their text captions."""

    def __init__(self, provider: str = "ollama", model: str | None = None):
        self._store = NumpyStore(provider=provider, model=model)
        self._images: dict[str, ImageDoc] = {}

    def add(self, images: list[ImageDoc]) -> None:
        chunks = [
            Chunk(img.caption, source=img.id, index=i, meta={"image_id": img.id, **img.meta})
            for i, img in enumerate(images)
        ]
        for img in images:
            self._images[img.id] = img
        self._store.add(chunks)

    def search(self, query: str, k: int = 4) -> list[tuple[ImageDoc, float]]:
        """Return the images whose captions best match the query."""
        hits: list[Hit] = self._store.search(query, k=k)
        return [(self._images[h.chunk.source], h.score) for h in hits]

    def __len__(self) -> int:
        return len(self._images)
