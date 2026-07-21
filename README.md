---
title: RGPV RAG Assistant
emoji: 🎓
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
license: mit
---

# RGPV RAG Assistant

An AI-powered Retrieval-Augmented Generation (RAG) assistant built for RGPV engineering students.

## Features

- 📚 Ask questions from your uploaded engineering notes
- 🔍 Hybrid Retrieval
  - Dense Search (Pinecone)
  - BM25 Sparse Search
  - Reciprocal Rank Fusion (RRF)
- 🎯 Cross-Encoder Reranking
- 💬 Multi-turn conversation memory
- 🛡️ Academic-domain moderation
- 🤖 Supports both Groq and Gemini user API keys

## Tech Stack

- Flask
- LlamaIndex
- Pinecone
- Hugging Face Embeddings (BAAI/bge-small-en-v1.5)
- BM25 (`rank_bm25`)
- CrossEncoder (`ms-marco-MiniLM-L-6-v2`)
- Gunicorn

## Environment Variables

The Space requires the following secrets:

- `PINECONE_API_KEY`
- `GROQ_API_KEY`
- `FLASK_SECRET_KEY`

## Notes

- The BM25 corpus is pre-generated and bundled with the application.
- Hybrid retrieval combines dense and sparse search using Reciprocal Rank Fusion before reranking.
- User-provided Groq or Gemini API keys are used for answer generation.
