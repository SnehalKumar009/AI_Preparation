---
title: Hybrid Retrieval and Reranking
tags: [bm25, keyword, hybrid, rrf, reranking, compression]
audience: engineers
---

# Hybrid Retrieval and Reranking

Dense vector search captures meaning but can miss exact tokens; keyword search
captures exact tokens but misses paraphrases. Hybrid retrieval combines both, and
reranking then sharpens the final order.

## Keyword search with BM25

BM25 is a sparse, term-frequency ranking function. It scores a document by how
often the query terms appear, dampened by document length and by how common each
term is across the corpus. It excels at exact matches: error codes, function
names, product SKUs, and rare proper nouns that embeddings often blur together.

## Combining dense and sparse

Hybrid search runs both retrievers and fuses their results. Reciprocal Rank
Fusion (RRF) is a simple, robust method: each result gets a score of one over a
constant plus its rank in each list, and the scores are summed. RRF needs no
score calibration between the two systems, which makes it a strong default.

## Reranking

A first-stage retriever favors recall — return a large candidate set cheaply. A
reranker then favors precision, scoring each candidate against the query with a
more expensive model. A cross-encoder reads the query and passage together for an
accurate relevance score. When you have no cross-encoder, an LLM can rerank by
scoring or ordering the candidates directly.

## Context compression

Even good passages contain filler. Context compression trims retrieved text down
to the sentences that actually matter before sending it to the generator. This
lowers cost, reduces distraction, and mitigates the lost-in-the-middle effect
where models overlook information buried in a long prompt.

## Citations and grounding

Grounded generation asks the model to answer only from the retrieved context and
to attach a source tag to each claim. Citations make answers verifiable and make
failures debuggable: if a claim has no supporting passage, it is likely a
hallucination and should be flagged.
