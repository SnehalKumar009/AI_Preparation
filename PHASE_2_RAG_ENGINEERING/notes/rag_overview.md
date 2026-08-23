---
title: Retrieval-Augmented Generation Overview
tags: [rag, retrieval, grounding]
audience: engineers
---

# Retrieval-Augmented Generation (RAG)

RAG combines a retriever with a generator. Instead of relying only on the
parameters learned during pretraining, the system fetches relevant text from an
external corpus at query time and conditions the language model on it. This
grounds answers in source material and lets you update knowledge by editing the
corpus rather than retraining the model.

## Why RAG exists

Large language models hallucinate when asked about facts outside their training
distribution, and their knowledge has a cutoff date. RAG addresses both problems:
it injects fresh, authoritative context into the prompt and gives the model
something concrete to cite. It is usually far cheaper than fine-tuning and much
easier to keep current.

## The core pipeline

A minimal RAG pipeline has four stages. First, ingestion parses raw documents
into clean text. Second, chunking splits that text into passages small enough to
embed and retrieve precisely. Third, indexing embeds each chunk into a vector and
stores it for similarity search. Fourth, at query time the retriever finds the
most relevant chunks, and the generator answers using them as context.

## Where quality comes from

Retrieval quality dominates end-to-end quality. If the right passage is never
retrieved, no amount of prompting will recover the answer. That is why serious
systems invest in chunking strategy, hybrid retrieval, reranking, and evaluation
rather than only tuning the final prompt.

## Common failure modes

Chunks that are too large dilute the relevant signal; chunks that are too small
lose context. Pure vector search misses exact keyword matches like error codes or
product names. Retrieved passages placed in the middle of a long prompt are often
ignored — the "lost in the middle" effect. Each of these has a dedicated
mitigation covered later in this phase.
