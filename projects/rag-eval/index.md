---
layout: default
title: Grounded Conversation RAG
description: Hybrid retrieval system with grounded answers, citations, abstention safeguards, evaluation benchmarks, and product-style delivery through FastAPI and Streamlit.
---
<div class="content-page" markdown="1">

# Grounded Conversation RAG

A production-style retrieval-augmented generation system for answering questions over internal-style knowledge sources such as policies, operations documents, and support conversations. The project combines lexical, dense, and hybrid retrieval with grounded answer generation, citations, abstention safeguards, evaluation benchmarks, and telemetry surfaced through both a FastAPI service and a Streamlit interface.

<div class="button-row project-actions">
  <a class="button" href="https://grounded-conversation-rag.streamlit.app" target="_blank" rel="noopener">Launch live Streamlit app</a>
  <a class="button-secondary" href="https://github.com/liamge/grounded-conversation-rag" target="_blank" rel="noopener">View GitHub repo</a>
</div>

<div class="metric-grid">
  <div class="metric-card">
    <div class="metric-value">4 retrieval modes</div>
    <p>TF-IDF, BM25, dense MiniLM, and hybrid fusion.</p>
  </div>
  <div class="metric-card">
    <div class="metric-value">2 delivery surfaces</div>
    <p>Interactive Streamlit demo plus FastAPI backend.</p>
  </div>
  <div class="metric-card">
    <div class="metric-value">Grounded outputs</div>
    <p>Citation-aware generation with explicit abstention behavior.</p>
  </div>
  <div class="metric-card">
    <div class="metric-value">Measured quality</div>
    <p>Recall@k, Precision@k, MRR, citation coverage, and evidence overlap.</p>
  </div>
</div>

## The problem

Teams increasingly want question-answering systems over internal knowledge, but naive RAG systems often break in predictable ways: retrieval brings back weak evidence, answers drift beyond the provided context, and it becomes difficult to explain whether the system is improving.

This project was built to demonstrate a more disciplined approach to retrieval and grounded generation. Instead of treating RAG as “embed documents and prompt an LLM,” the system explicitly separates ingestion, chunking, retrieval, reranking, context assembly, answer generation, and evaluation.

## What I built

This repository implements an end-to-end grounded RAG workflow with:

- deterministic ingestion of `.txt`, `.md`, `.json`, and `.jsonl` documents
- sentence-aware chunking with overlap and stable chunk IDs
- multiple retrieval strategies: TF-IDF, BM25, dense retrieval, and hybrid fusion
- optional reranking and diversity controls
- grounded generation with citation-enforcing prompts
- abstention behavior when evidence is insufficient
- FastAPI endpoints for querying, health checks, index rebuilds, and metrics
- a Streamlit application designed for interactive demos and portfolio screenshots
- evaluation tooling for retrieval and grounding quality
- telemetry, reports, tests, Docker support, and CI

## Why this project is stronger than a basic RAG demo

Many portfolio RAG projects only demonstrate vector search plus a prompt template. This one goes deeper in the parts that actually matter in production-style systems:

### Retrieval engineering

The core technical depth is in the retrieval layer. The repo supports:

- **TF-IDF** as a sparse lexical baseline
- **BM25** for stronger keyword retrieval
- **Dense retrieval** using sentence-transformer embeddings
- **Hybrid fusion** to combine lexical and semantic signals
- optional **FAISS-backed artifact caching**
- optional **reranking** and **diversity filtering**

That makes the project a retrieval system, not just an LLM wrapper.

### Grounded answer generation

The generation layer is designed to keep answers tied to evidence. It uses:

- context budgets
- citation-aware prompts
- explicit abstention behavior
- deterministic fallback generation when API keys are unavailable

That is important for credibility because it shows the system is built for groundedness and repeatability, not just convenience.

### Measurement and evaluation

The repo includes a real evaluation surface, not placeholder claims. It measures:

- Recall@k
- Precision@k
- Mean Reciprocal Rank (MRR)
- citation coverage
- evidence overlap
- abstention correctness

This makes it possible to compare retrieval strategies and answer quality using structured outputs rather than anecdotal examples.

## System architecture

<div class="architecture-block">
  <strong>Raw documents</strong> → ingestion → chunking → indexing (TF-IDF / BM25 / dense / hybrid) → optional reranking → context assembly → grounded generation → citations / abstention → telemetry and reports
</div>

The architecture is intentionally modular so that each stage can be tested and improved independently.

## Product surfaces

## 1. Streamlit demo

The Streamlit application is one of the strongest portfolio assets in the repo. It includes:

- a query interface
- retrieval mode selection
- rerank toggle
- chunk inspection
- latency and telemetry views
- benchmark and evaluation-oriented displays

This makes the project easier to explain visually and better suited for public case-study screenshots.

## 2. FastAPI service

The FastAPI backend exposes the retrieval pipeline through a service layer with endpoints such as:

- `/health`
- `/query`
- `/index/rebuild`
- `/metrics`

The app also includes startup indexing, request IDs, timing middleware, structured error handling, and telemetry aggregation.

That gives the project a more realistic deployment shape than a notebook-only prototype.

## Evaluation and observability

The project includes evaluation and monitoring components that support more rigorous benchmarking over time.

Examples of what is tracked:

- retrieval relevance
- ranking quality
- citation usage
- answer grounding
- abstention behavior
- query latency
- aggregate telemetry summaries written to reports

This is one of the strongest parts of the repo because it shows an understanding that RAG systems need measurement, not just generation.

## Technologies used

Python  
FastAPI  
Streamlit  
sentence-transformers  
FAISS  
scikit-learn  
BM25 / sparse retrieval methods  
Docker  
Pytest  
GitHub Actions

## What this project demonstrates

This case study is strongest as evidence of:

- retrieval systems engineering
- grounded LLM application design
- evaluation-aware RAG development
- API and UI delivery for technical products
- production-minded packaging with tests, config, and containerization

## Future improvements

The next upgrades that would make this even stronger as a public portfolio case study are:

- benchmark result tables or charts added directly to the page
- screenshots of the Streamlit interface
- one architecture diagram image
- a short deployment section once a public demo is hosted
- side-by-side comparisons of retrieval modes on the same query set

</div>
