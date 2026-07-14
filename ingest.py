import os
import json
import chromadb  # remove once fully migrated; harmless if left, but can delete this line
from pinecone import Pinecone
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.pinecone import PineconeVectorStore

from config import (
    PDF_DIR,
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    MANIFEST_PATH,
    EMBED_MODEL_NAME,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
)


def load_manifest() -> set:
    if not os.path.exists(MANIFEST_PATH):
        return set()
    with open(MANIFEST_PATH, "r") as f:
        return set(json.load(f))


def save_manifest(files: set):
    os.makedirs(os.path.dirname(MANIFEST_PATH), exist_ok=True)
    with open(MANIFEST_PATH, "w") as f:
        json.dump(sorted(files), f, indent=2)


def find_all_pdfs(root: str) -> set:
    """Recursively find PDFs, return relative paths from root (so subject
    folders are preserved and filenames don't collide across subjects)."""
    found = set()
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().endswith(".pdf"):
                rel = os.path.relpath(os.path.join(dirpath, fn), root)
                found.add(rel)
    return found


def load_documents():
    print(f"Loading PDFs from: {PDF_DIR} (recursive)")
    documents = SimpleDirectoryReader(
        input_dir=PDF_DIR,
        required_exts=[".pdf"],
        filename_as_id=True,
        recursive=True,
    ).load_data()

    for doc in documents:
        file_path = doc.metadata.get("file_path", "")
        filename = os.path.basename(file_path)
        rel_path = os.path.relpath(file_path, PDF_DIR)
        subject = os.path.basename(os.path.dirname(file_path)) or "general"

        doc.metadata["filename"] = filename
        doc.metadata["rel_path"] = rel_path
        doc.metadata["subject"] = subject
        doc.metadata["page_no"] = doc.metadata.get("page_label", "?")

    print(f"Loaded {len(documents)} pages.")
    return documents


def build_index(documents, vector_store):
    print(f"Loading embed model: {EMBED_MODEL_NAME}")
    embed_model = HuggingFaceEmbedding(
        model_name=EMBED_MODEL_NAME,
        embed_batch_size=32,
    )

    splitter = SentenceSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    print("Chunking, embedding, upserting to Pinecone...")
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        embed_model=embed_model,
        transformations=[splitter],
        show_progress=True,
    )

    print("Ingestion complete.")
    return index


if __name__ == "__main__":
    if not os.path.isdir(PDF_DIR):
        raise SystemExit(f"PDF_DIR '{PDF_DIR}' does not exist. Create it and add PDFs first.")

    if not PINECONE_API_KEY:
        raise SystemExit("PINECONE_API_KEY not set. Add it to your .env / HF Space secrets.")

    pc = Pinecone(api_key=PINECONE_API_KEY)
    pinecone_index = pc.Index(PINECONE_INDEX_NAME)
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)

    existing_files = load_manifest()
    all_files = find_all_pdfs(PDF_DIR)
    new_files = all_files - existing_files

    if not new_files:
        print(f"No new PDFs found. {len(existing_files)} files already ingested.")
    else:
        print(f"New files detected ({len(new_files)}): {sorted(new_files)}")
        docs = load_documents()
        new_docs = [d for d in docs if d.metadata.get("rel_path") in new_files]
        print(f"Ingesting {len(new_docs)} pages from {len(new_files)} new files.")
        build_index(new_docs, vector_store)

        existing_files |= new_files
        save_manifest(existing_files)
        print(f"Manifest updated: {len(existing_files)} files tracked.")