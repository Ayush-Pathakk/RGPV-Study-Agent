"""
Hybrid (dense + BM25) retrieval support.

Kept in its own module on purpose: dense retrieval (query.py) and this file
have zero import dependency on each other's internals. If BM25 needs to be
swapped for a different sparse method later, only this file changes.

Corpus persistence: chunk text/metadata is appended to a local JSONL file at
ingestion time (see ingest.py). This module only ever *reads* that file — it
never talks to Pinecone directly.
"""
import json
import os
import re
from rank_bm25 import BM25Okapi


def _tokenize(text: str) -> list[str]:
    """Simple, dependency-free tokenizer: lowercase, strip punctuation, split.
    Good enough for keyword matching on engineering notes. No stemming —
    add nltk/snowball later only if recall on term variants proves weak."""
    return re.findall(r"[a-z0-9]+", text.lower())


def append_corpus_records(records: list[dict], corpus_path: str) -> None:
    """Append new chunk records (called once per newly-ingested file).
    Never rewrites existing lines — safe to call incrementally, matches
    the manifest pattern already used for Pinecone ingestion."""
    os.makedirs(os.path.dirname(corpus_path), exist_ok=True)
    with open(corpus_path, "a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_corpus(corpus_path: str) -> list[dict]:
    """Returns [] if the file doesn't exist yet — callers must treat that
    as 'BM25 unavailable, dense-only', not as an error."""
    if not os.path.exists(corpus_path):
        return []
    records = []
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


class BM25Index:
    """Wraps rank_bm25 with the id<->record bookkeeping needed for fusion.
    Rebuilt once at app startup (in-memory, cheap at this corpus size —
    reconsider only if the corpus grows past ~50k chunks)."""

    def __init__(self, corpus_path: str):
        self.records = load_corpus(corpus_path)
        self.available = len(self.records) > 0
        if self.available:
            tokenized = [_tokenize(r["text"]) for r in self.records]
            self._bm25 = BM25Okapi(tokenized)
        else:
            self._bm25 = None

    def search(self, query: str, top_k: int) -> list[dict]:
        if not self.available:
            return []
        scores = self._bm25.get_scores(_tokenize(query))
        ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [self.records[i] for i in ranked_idx[:top_k] if scores[i] > 0]


def reciprocal_rank_fusion(ranked_lists: list[list[str]], k: int = 60) -> list[str]:
    """Standard RRF: score(id) = sum(1 / (k + rank)) across all lists it
    appears in. Returns ids ordered best-first. k=60 is the widely-used
    default from the original RRF paper — no dataset-specific tuning needed."""
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.keys(), key=lambda i: scores[i], reverse=True)