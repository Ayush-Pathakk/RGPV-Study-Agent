import os
from dotenv import load_dotenv

# Loads variables from a .env file in the project root into the environment.
load_dotenv()

PDF_DIR = "data/pdfs"
CHROMA_DIR = "data/chroma_db"
COLLECTION_NAME = "rgpv_notes"

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL_NAME = "llama-3.3-70b-versatile"

CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
MAX_NEW_TOKEN = 400
MIN_NEW_TOKEN = 1
TOP_P = 1
TOP_K = 3