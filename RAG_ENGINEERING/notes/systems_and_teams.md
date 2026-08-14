---
title: RAG Systems Field Notes
tags: [systems, case-study, architecture]
audience: engineers
---

# RAG Systems Field Notes

These notes describe how a few internal systems fit together. The facts are
deliberately spread across sections so a single question often needs more than one
passage to answer — the setup that multi-hop retrieval and graph RAG are built for.

## Atlas retrieval service

Atlas is the retrieval service that powers the Helpdesk Assistant. It indexes the
support knowledge base, embeds passages with the nomic-embed-text model, and stores
them in a Chroma vector database. Atlas was built by the Search Platform team and
calls the Beacon reranker before returning results.

## Helpdesk Assistant

The Helpdesk Assistant answers customer support questions for external users. It is
powered by the Atlas retrieval service and is maintained by the Support Engineering
team. It relies on hybrid retrieval so that exact error codes are matched alongside
paraphrased questions.

## Beacon reranker

Beacon is the reranking component that Atlas calls to reorder its top candidates
before they reach the generator. Beacon is a cross-encoder trained on labeled
support tickets. It was built by the Search Platform team.

## Search Platform team

The Search Platform team owns both Atlas and Beacon. The team is led by Priya Nair
and focuses on retrieval quality, indexing, and reranking across the company. It
publishes the shared embedding guidelines that other teams follow.
