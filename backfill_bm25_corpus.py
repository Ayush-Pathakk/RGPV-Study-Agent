"""
Run this ONCE, before deploying hybrid search, to backfill BM25 corpus
records for the 8,122 vectors already in Pinecone from before this feature
existed. Without this, hybrid search would silently only cover documents
ingested after this point — old content stays dense-only forever.

Safe to re-run: append_corpus_records always appends, so if you run this
twice you'll get duplicate lines. If that happens, just delete
data/bm25_corpus.jsonl and re-run once.

Usage: python backfill_bm25_corpus.py
"""
from pinecone import Pinecone
from config import PINECONE_API_KEY, PINECONE_INDEX_NAME, CORPUS_PATH
from hybrid_retrieval import append_corpus_records

FETCH_BATCH = 100  # Pinecone fetch() batch size — conservative, avoids payload limits


def backfill():
    if not PINECONE_API_KEY:
        raise SystemExit("PINECONE_API_KEY not set.")

    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX_NAME)

    stats = index.describe_index_stats()
    total = stats.total_vector_count
    print(f"Pinecone reports {total} vectors. Backfilling BM25 corpus...")

    all_ids = []

    for page in index.list():
        all_ids.extend(item.id for item in page.vectors)

    print(f"Collected {len(all_ids)} vector ids.")

    written = 0
    for i in range(0, len(all_ids), FETCH_BATCH):
        batch_ids = all_ids[i:i + FETCH_BATCH]
        fetched = index.fetch(ids=batch_ids)
        records = []
        for vec_id, vec in fetched.vectors.items():
            md = vec.metadata or {}
            text = md.get("text", "")
            if not text:
                print(f"WARNING: vector {vec_id} has no 'text' in metadata — skipped.")
                continue
            records.append({
                "id": vec_id,
                "text": text,
                "filename": md.get("filename", "?"),
                "subject": md.get("subject", "?"),
                "page_no": md.get("page_no", "?"),
            })
        append_corpus_records(records, CORPUS_PATH)
        written += len(records)
        print(f"  {written}/{len(all_ids)} backfilled...")

    print(f"Done. {written} records written to {CORPUS_PATH}.")


if __name__ == "__main__":
    backfill()