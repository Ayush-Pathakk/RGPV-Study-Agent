import os
from dotenv import load_dotenv

load_dotenv()

PDF_DIR = "data/pdfs"

# Pinecone
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = "rgpv-data"

MANIFEST_PATH = "data/ingested_files.json"

EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL_NAME = "openai/gpt-oss-120b"
SAFEGUARD_MODEL_NAME = "openai/gpt-oss-safeguard-20b"

CHUNK_SIZE = 512
CHUNK_OVERLAP = 128

RETRIEVAL_TOP_K = 10
RERANKER_TOP_N = 3
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANKER_THRESHOLD = 0.5

MAX_NEW_TOKEN = 2048
MIN_NEW_TOKEN = 1
TOP_P = 1
TOP_K = 3