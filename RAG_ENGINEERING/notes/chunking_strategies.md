---
title: Chunking Strategies
tags: [chunking, chunk-size, overlap, parent-child]
audience: engineers
---

# Chunking Strategies

Chunking splits documents into passages that are indexed and retrieved. It is the
most underrated lever in RAG: the wrong chunking silently caps retrieval quality
no matter how good the embedding model or the reranker is.

## Fixed-size chunking

The simplest approach slices text into windows of a fixed number of tokens or
words, often with a small overlap between neighbors. It is fast and predictable
but ignores document structure, so it can cut sentences or ideas in half.

## Chunk size trade-offs

Small chunks give precise retrieval and tight context, but a single chunk may
lack the surrounding information needed to answer. Large chunks preserve context
but dilute the relevant signal and waste tokens. A common starting range is 200
to 500 tokens, tuned against an evaluation set rather than guessed.

## Overlap

Overlap repeats a slice of text between adjacent chunks so that an idea straddling
a boundary still appears whole in at least one chunk. Typical overlap is ten to
twenty percent of the chunk size. Too much overlap inflates the index and returns
near-duplicate results.

## Structure-aware and semantic chunking

Recursive chunking splits on natural separators — paragraphs, then sentences —
before falling back to raw character windows, keeping related text together.
Semantic chunking goes further: it embeds sentences and starts a new chunk where
the meaning shifts, so each chunk is topically coherent.

## Parent-child chunking

Parent-child (small-to-big) indexing embeds small child chunks for precise
matching but returns their larger parent passage to the generator. You get the
retrieval precision of small chunks with the context of large ones. The
sentence-window variant retrieves a single sentence and expands to a window of
neighboring sentences at answer time.
