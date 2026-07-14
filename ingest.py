import os
print("1")
import chromadb
print("2")
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
print("3")
from llama_index.core.node_parser import SentenceSplitter
print("4")
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
print("5")
from llama_index.vector_stores.chroma import ChromaVectorStore
print("6")

from config import (
    PDF_DIR,
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBED_MODEL_NAME,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
)

def load_documents():
    print(f"Loading PDFs from: {PDF_DIR}")
    documents = SimpleDirectoryReader(
        input_dir=PDF_DIR,
        required_exts=[".pdf"],
        filename_as_id=True,
    ).load_data()

    for doc in documents:
        filename = os.path.basename(doc.metadata.get("file_path", "unknown"))
        subject  = os.path.splitext(filename)[0]
        doc.metadata["filename"] = filename
        doc.metadata["subject"]  = subject
        doc.metadata["page_no"]  = doc.metadata.get("page_label", "?")

    print(f"Loaded {len(documents)} pages.")
    return documents

def build_index(documents):
    print(f"Loading embed model: {EMBED_MODEL_NAME}")
    embed_model = HuggingFaceEmbedding(
        model_name=EMBED_MODEL_NAME,
        embed_batch_size=32,
    )

    splitter = SentenceSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    chroma_client     = chromadb.PersistentClient(path=CHROMA_DIR)
    chroma_collection = chroma_client.get_or_create_collection(COLLECTION_NAME)
    vector_store      = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context   = StorageContext.from_defaults(vector_store=vector_store)

    print("Chunking, embedding, storing...")
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        embed_model=embed_model,
        transformations=[splitter],
        show_progress=True,
    )

    print(f"Ingestion complete. Total chunks: {chroma_collection.count()}")
    return index

if __name__ == "__main__":
    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection    = chroma_client.get_or_create_collection(COLLECTION_NAME)

    existing_files = set()
    if collection.count() > 0:
        results = collection.get(include=["metadatas"])
        for meta in results["metadatas"]:
            if meta and "filename" in meta:
                existing_files.add(meta["filename"])

    all_files = set(
        f for f in os.listdir(PDF_DIR) if f.endswith(".pdf")
    )
    new_files = all_files - existing_files

    if not new_files:
        print(f"No new PDFs found. Chroma has {collection.count()} chunks.")
    else:
        print(f"New files detected: {new_files}")
        docs = load_documents()
        new_docs = [
            d for d in docs
            if os.path.basename(d.metadata.get("file_path", "")) in new_files
        ]
        print(f"Ingesting {len(new_docs)} pages from {len(new_files)} new files.")
        build_index(new_docs)