"""
One-time backfill for BM25 corpus from existing Pinecone vectors.
Safe to delete after successful execution.
"""

import json
from pinecone import Pinecone
from config import PINECONE_API_KEY, PINECONE_INDEX_NAME, CORPUS_PATH
from hybrid_retrieval import append_corpus_records

FETCH_BATCH = 100


def extract_text(metadata: dict):
    """Try all known LlamaIndex metadata layouts."""

    if not metadata:
        return None

    # Direct text
    for key in (
        "text",
        "chunk_text",
        "content",
        "page_content",
        "_node_content",
        "__node_content__",
    ):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            # __node_content__ is often JSON
            if key == "__node_content__":
                try:
                    obj = json.loads(value)
                    return (
                        obj.get("text")
                        or obj.get("content")
                        or value
                    )
                except Exception:
                    return value
            return value

    # Last resort: inspect every string field
    for value in metadata.values():
        if isinstance(value, str):
            if len(value) > 100:
                return value

    return None


def backfill():

    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX_NAME)

    stats = index.describe_index_stats()

    print(f"Vectors in Pinecone: {stats.total_vector_count}")

    all_ids = []

    for page in index.list():

        # SDK v5
        if hasattr(page, "vectors"):
            all_ids.extend(item.id for item in page.vectors)

        # Older SDK
        else:
            all_ids.extend(page)

    print(f"Collected {len(all_ids)} ids")

    written = 0

    for start in range(0, len(all_ids), FETCH_BATCH):

        ids = all_ids[start:start + FETCH_BATCH]

        fetched = index.fetch(ids=ids)

        records = []

        for vec_id, vec in fetched.vectors.items():

            md = vec.metadata or {}

            text = extract_text(md)

            if not text:
                print(f"Skipping {vec_id} (no text found)")
                continue

            records.append(
                {
                    "id": vec_id,
                    "text": text,
                    "filename": md.get("filename", ""),
                    "subject": md.get("subject", ""),
                    "page_no": md.get("page_no", ""),
                }
            )

        append_corpus_records(records, CORPUS_PATH)

        written += len(records)

        print(
            f"{written}/{len(all_ids)} written"
        )

    print()
    print("=" * 60)
    print(f"Finished. {written} records written.")
    print("=" * 60)


if __name__ == "__main__":
    backfill()