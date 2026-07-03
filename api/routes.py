"""
Flask API Routes.

Endpoints:
  POST /chat          — RAG question answering
  POST /upload-pdf    — Ingest a new PDF into the knowledge base
  GET  /health        — Service health check
  GET  /models        — List available models
  POST /switch-model  — Change active LLM at runtime
  GET  /sources       — List ingested PDF sources
"""

import logging
import time

from flask import Blueprint, request, jsonify, current_app
from pydantic import ValidationError

import config
from api.schemas import (
    ChatRequest,
    SwitchModelRequest,
)

logger = logging.getLogger(__name__)

bp = Blueprint("api", __name__)

APP_VERSION = "1.0.0"


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _get_rag_pipeline():
    """Retrieve the shared RAG pipeline components from the app context."""
    return (
        current_app.retriever,
        current_app.rag_generator,
        current_app.registry,
        current_app.vector_store,
    )


def _json_error(message: str, detail: str = None, status: int = 400):
    body = {"error": message}
    if detail:
        body["detail"] = detail
    return jsonify(body), status


# ─── POST /chat ───────────────────────────────────────────────────────────────


@bp.route("/chat", methods=["POST"])
def chat():
    """
    Main RAG endpoint.

    Body (JSON):
      {
        "question": "What is covered in chapter 2?",
        "model": "llama-3.3-70b-versatile",  // optional
        "top_k": 5,                          // optional
        "rerank_top_k": 3,                   // optional
        "use_reranking": true                // optional
      }
    """
    retriever, rag_gen, registry, vector_store = _get_rag_pipeline()

    try:
        payload = ChatRequest.model_validate(request.get_json(force=True) or {})
    except ValidationError as exc:
        return _json_error("Invalid request", str(exc), 422)

    if rag_gen is None:
        return _json_error(
            "LLM generator is not configured. Set a valid GROQ_API_KEY in .env.",
            status=503,
        )

    if vector_store.count() == 0:
        return _json_error(
            "Knowledge base is empty. Upload a PDF first via POST /upload-pdf.",
            status=503,
        )

    t0 = time.perf_counter()
    try:
        docs = retriever.retrieve_documents(
            query=payload.question,
            top_k=payload.top_k or config.TOP_K,
            rerank_top_k=payload.rerank_top_k or config.RERANK_TOP_K,
            use_reranking=payload.use_reranking
            if payload.use_reranking is not None
            else config.ENABLE_RERANKING,
        )

        if not docs:
            result = rag_gen.answer_without_context(
                payload.question, model_name=payload.model
            )
        else:
            result = rag_gen.generate_answer(
                question=payload.question,
                documents=docs,
                model_name=payload.model,
            )

        elapsed = round(time.perf_counter() - t0, 3)
        result["latency_s"] = elapsed
        logger.info(
            f"[/chat] Q='{payload.question[:60]}…' | "
            f"docs={result['retrieved_count']} | latency={elapsed}s"
        )
        return jsonify(result), 200

    except Exception as exc:
        logger.exception(f"[/chat] Unexpected error: {exc}")
        return _json_error("Generation failed", str(exc), 500)


# ─── POST /upload-pdf ─────────────────────────────────────────────────────────


@bp.route("/upload-pdf", methods=["POST"])
def upload_pdf():
    """
    Upload and ingest a PDF into the knowledge base.

    Multipart form data:
      file: <PDF binary>
    """
    retriever, rag_gen, registry, vector_store = _get_rag_pipeline()
    ingestion = current_app.ingestion
    chunker = current_app.chunker
    embedding_model = current_app.embedding_model

    if "file" not in request.files:
        return _json_error("No file part in request. Use multipart field 'file'.")

    file = request.files["file"]
    if file.filename == "":
        return _json_error("Empty filename.")
    if not file.filename.lower().endswith(".pdf"):
        return _json_error("Only PDF files are accepted.")

    save_path = config.DATA_DIR / file.filename
    file.save(str(save_path))
    logger.info(f"PDF saved to {save_path}")

    try:
        pages = ingestion.process_pdf(str(save_path))
        if not pages:
            return _json_error("PDF contained no extractable text.", status=422)

        chunks = chunker.chunk_text(pages)
        if not chunks:
            return _json_error("No chunks produced from PDF.", status=422)

        texts = [c["text"] for c in chunks]
        embeddings = embedding_model.embed_documents(texts)

        added = vector_store.add_documents(chunks, embeddings)

        # Update BM25 index if the retriever supports it
        if hasattr(retriever, "update_corpus"):
            retriever.update_corpus(chunks)

        logger.info(
            f"Ingested '{file.filename}': {len(pages)} pages, "
            f"{len(chunks)} chunks, {added} vectors stored"
        )
        return (
            jsonify(
                {
                    "filename": file.filename,
                    "pages_processed": len(pages),
                    "chunks_created": len(chunks),
                    "message": f"Successfully ingested {file.filename}",
                }
            ),
            201,
        )

    except Exception as exc:
        logger.exception(f"[/upload-pdf] Failed to ingest {file.filename}: {exc}")
        save_path.unlink(missing_ok=True)
        return _json_error("Ingestion failed", str(exc), 500)


# ─── GET /health ──────────────────────────────────────────────────────────────


@bp.route("/health", methods=["GET"])
def health():
    """Service health check."""
    retriever, rag_gen, registry, vector_store = _get_rag_pipeline()

    try:
        from models.qwen_loader import detect_available_gpu
        gpu_info = detect_available_gpu()
    except Exception:
        gpu_info = {}

    return (
        jsonify(
            {
                "status": "ok",
                "version": APP_VERSION,
                "active_model": registry.current_model_name,
                "active_embedding": current_app.embedding_model.model_name,
                "vector_store_backend": config.VECTOR_STORE_BACKEND,
                "document_count": vector_store.count(),
                "gpu_info": gpu_info,
            }
        ),
        200,
    )


# ─── GET /models ──────────────────────────────────────────────────────────────


@bp.route("/models", methods=["GET"])
def list_models():
    """List all available LLM models and their metadata."""
    retriever, rag_gen, registry, vector_store = _get_rag_pipeline()

    qwen_loaded = bool(
        rag_gen and rag_gen._qwen_generator and rag_gen._qwen_generator.is_loaded
    )

    return (
        jsonify(
            {
                "active_model": registry.current_model_name,
                "models": registry.list_models(qwen_loaded=qwen_loaded),
                "embedding_models": list(config.EMBEDDING_MODELS.keys()),
                "active_embedding": current_app.embedding_model.model_name,
            }
        ),
        200,
    )


# ─── POST /switch-model ───────────────────────────────────────────────────────


@bp.route("/switch-model", methods=["POST"])
def switch_model():
    """
    Switch the active LLM model at runtime.

    Body (JSON):
      { "model": "llama-3.1-8b-instant" }
    """
    retriever, rag_gen, registry, vector_store = _get_rag_pipeline()

    try:
        payload = SwitchModelRequest.model_validate(request.get_json(force=True) or {})
    except ValidationError as exc:
        return _json_error("Invalid request", str(exc), 422)

    try:
        result = registry.switch_model(payload.model)
        return (
            jsonify(
                {
                    "previous_model": result["previous_model"],
                    "current_model": result["current_model"],
                    "message": f"Active model switched to '{payload.model}'",
                }
            ),
            200,
        )
    except Exception as exc:
        logger.exception(f"[/switch-model] Failed: {exc}")
        return _json_error("Model switch failed", str(exc), 500)


# ─── GET /sources ─────────────────────────────────────────────────────────────


@bp.route("/sources", methods=["GET"])
def list_sources():
    """Return the list of PDF sources currently in the knowledge base."""
    retriever, rag_gen, registry, vector_store = _get_rag_pipeline()
    sources = vector_store.list_sources()
    return jsonify({"sources": sources, "count": len(sources)}), 200
