"""
Application entrypoint.

Wires together ingestion, chunking, embeddings, vector store, BM25,
retrieval, the model registry and the RAG generator, then attaches
them to the Flask app so routes can reach them via `current_app`.
"""

import logging
import os

from flask import Flask, render_template

import config
from api.routes import bp as api_bp
from rag.ingestion import PDFIngestion
from rag.chunking import TextChunker
from rag.embeddings import EmbeddingModel
from rag.retrieval import HybridRetriever
from vectorDB.chroma_store import ChromaStore
from vectorDB.BM25_DB import BM25Store
from models.registry import ModelRegistry
from models.generator import RAGGenerator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger(__name__)


def create_app() -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024

    # -------------------------------
    # Initialize RAG pipeline
    # -------------------------------
    app.ingestion = PDFIngestion()
    app.chunker = TextChunker()
    app.embedding_model = EmbeddingModel()
    app.vector_store = ChromaStore()
    app.bm25_store = BM25Store()

    app.retriever = HybridRetriever(
        chroma_store=app.vector_store,
        bm25_store=app.bm25_store,
        embedding_model=app.embedding_model,
    )

    app.registry = ModelRegistry()
    app.rag_generator = RAGGenerator(registry=app.registry)

    if not config.GROQ_API_KEY or config.GROQ_API_KEY == "your_groq_api_key":
        logger.warning(
            "GROQ_API_KEY is not set — Groq models will be unavailable. "
            "Qwen (local) will still work."
        )

    # -------------------------------
    # Web UI
    # -------------------------------
    @app.route("/")
    def home():
        return render_template("index.html")

    # -------------------------------
    # API Routes
    # -------------------------------
    app.register_blueprint(api_bp)

    logger.info("RAG Flask app initialized.")

    return app


app = create_app()


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8080)),
        debug=False,
    )