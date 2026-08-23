---
title: Vector Search and Similarity
tags: [embeddings, cosine, ann, vector-database]
audience: engineers
---

# Vector Search and Similarity

Vector search retrieves text by meaning rather than by exact words. Each passage
is mapped to a dense embedding, and a query is embedded the same way. The
passages whose vectors are closest to the query vector are returned as the most
relevant results.

## Embeddings

An embedding is a fixed-length list of numbers produced by a model such as
`nomic-embed-text` or OpenAI `text-embedding-3-small`. Texts with similar meaning
land near each other in the vector space. Typical dimensions range from a few
hundred to a few thousand; higher dimensions can capture more nuance but cost more
to store and compare.

## Cosine similarity

Cosine similarity measures the angle between two vectors, ignoring their
magnitude. It ranges from -1 (opposite) through 0 (unrelated) to 1 (identical
direction). If you L2-normalize every vector to unit length first, cosine
similarity is simply the dot product, which makes retrieval a single fast matrix
multiplication.

## Approximate nearest neighbors

Scanning every vector (exact search) is accurate but slow at scale. Approximate
nearest neighbor algorithms trade a little recall for large speedups. HNSW builds
a navigable small-world graph and is the default in many vector databases. IVF
partitions vectors into clusters and only searches the nearest clusters. Product
quantization compresses vectors to shrink memory. Chroma, FAISS, and Pinecone all
build on these ideas.

## Vector databases

A vector database stores embeddings alongside metadata and offers persistence,
filtering, and fast approximate search. Chroma is a lightweight local option that
keeps an on-disk index so you do not re-embed a corpus on every run. Metadata
filters let you restrict search to, say, a single document, language, or date
range before ranking by similarity.
