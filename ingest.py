print("STARTING INGEST")

import os
print("1")
import chromadb
from llama_index.readers.file import PDFReader
print("2")
from llama_index.core import SimpleDirectoryReader , VectorStoreIndex , StorageContext
print("3")

from llama_index.core.node_parser import SentenceSplitter
print("4")
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

print("5")
from llama_index.vector_stores.chroma import ChromaVectorStore
print("6")

from config import(
    PDF_DIR,
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBED_MODEL_NAME,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
)

print("7")

def load_documents():
    """Load all PDF files from the data/pdfs directory."""
    print(f"Loading PDFs from: {PDF_DIR}")
    documents = SimpleDirectoryReader(
        input_dir=PDF_DIR,
        required_exts=[".pdf"],
        filename_as_id=True,
    ).load_data()
    print(f"Loaded {len(documents)} document pages.")
    return documents

def build_index(documents):
    """chunk documents , embed them and store them in chromadb"""
    print(f"load embedding model:{EMBED_MODEL_NAME} ")
    embed_model = HuggingFaceEmbedding(model_name = EMBED_MODEL_NAME)
    splitter = SentenceSplitter(chunk_size=CHUNK_SIZE,chunk_overlap= CHUNK_OVERLAP, separator= " ")
    print(f"connecting to chromadb at : {CHROMA_DIR}")
    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    chroma_collection = chroma_client.get_or_create_collection(COLLECTION_NAME)


    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    print("chunking , embedding , storing: this may take a min")
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        embed_model=embed_model,
        transformations=[splitter],
    )

    print("ingestion complete")
    return index

if __name__ == "__main__":

    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = chroma_client.get_or_create_collection(COLLECTION_NAME)

    if collection.count() > 0:
        print(f"Chroma already has {collection.count()} chunks. Skipping ingestion.")
    else:
        docs = load_documents()
        build_index(docs)

print("finished")