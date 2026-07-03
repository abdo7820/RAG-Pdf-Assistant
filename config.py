from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ─── Application ────────────────────────────────────────────────────────────
DEBUG = os.getenv("DEBUG", "False").lower() == "true"

# ─── Groq ───────────────────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ─── Qwen (local, via Hugging Face transformers) ───────────────────────────
QWEN_MODEL = os.getenv("QWEN_MODEL", "Qwen/Qwen3-8B")
QWEN_DEVICE = os.getenv("QWEN_DEVICE", "auto")  # "auto" | "cpu" | "cuda"
QWEN_MAX_NEW_TOKENS = int(os.getenv("QWEN_MAX_NEW_TOKENS", 1024))

# ─── Model provider / registry ─────────────────────────────────────────────
# Default provider used when no model is explicitly requested.
MODEL_PROVIDER = os.getenv("MODEL_PROVIDER", "groq")

DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "llama-3.3-70b-versatile")

# Every model the app is allowed to switch to at runtime.
# "provider" determines which backend RAGGenerator dispatches to:
#   "groq" -> Groq hosted API call
#   "qwen" -> local transformers inference
LLM_MODELS = {
    "llama-3.3-70b-versatile": {
        "provider": "groq",
        "description": "Meta Llama 3.3 70B — strong general-purpose model, served via Groq.",
        "max_tokens": 32768,
        "context_window": 128000,
    },
    "llama-3.1-8b-instant": {
        "provider": "groq",
        "description": "Meta Llama 3.1 8B — fast, lightweight model, served via Groq.",
        "max_tokens": 8192,
        "context_window": 128000,
    },
    "qwen3-8b": {
        "provider": "qwen",
        "description": f"Qwen3 8B ({QWEN_MODEL}) — runs locally via transformers.",
        "max_tokens": QWEN_MAX_NEW_TOKENS,
        "context_window": 32768,
    },
}

# ─── Embeddings ─────────────────────────────────────────────────────────────
EMBEDDING_MODELS = {
    "bge-base": "BAAI/bge-base-en-v1.5",
    "bge-small": "BAAI/bge-small-en-v1.5",
}
DEFAULT_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL_KEY", "bge-base")

# ─── Vector store ───────────────────────────────────────────────────────────
VECTOR_STORE_BACKEND = os.getenv("VECTOR_STORE_BACKEND", "chroma")
CHROMA_PATH = str(BASE_DIR / "vectorDB" / "db" / "chroma")
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "documents")

# ─── Chunking ───────────────────────────────────────────────────────────────
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 500))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))

# ─── Retrieval ──────────────────────────────────────────────────────────────
TOP_K = int(os.getenv("TOP_K", 5))
RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", 3))
ENABLE_RERANKING = os.getenv("ENABLE_RERANKING", "False").lower() == "true"
HYBRID_ALPHA = float(os.getenv("HYBRID_ALPHA", 0.5))  # weight for vector vs bm25

# Kept for backwards compatibility with any code importing the old Config class.
class Config:
    DEBUG = DEBUG
    GROQ_API_KEY = GROQ_API_KEY
    GROQ_MODEL = DEFAULT_MODEL
    QWEN_MODEL = QWEN_MODEL
    MODEL_PROVIDER = MODEL_PROVIDER
    EMBEDDING_MODEL = EMBEDDING_MODELS[DEFAULT_EMBEDDING_MODEL]
    CHROMA_PATH = CHROMA_PATH
    CHUNK_SIZE = CHUNK_SIZE
    CHUNK_OVERLAP = CHUNK_OVERLAP
    TOP_K = TOP_K
    HYBRID_ALPHA = HYBRID_ALPHA
