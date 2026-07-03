"""
API Schemas.

Pydantic v2 models for strict validation and auto-documentation.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


# ─── Request schemas ──────────────────────────────────────────────────────────


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=4096, description="User question")
    model: Optional[str] = Field(None, description="Override the active LLM model")
    top_k: Optional[int] = Field(None, ge=1, le=20, description="Number of chunks to retrieve")
    rerank_top_k: Optional[int] = Field(None, ge=1, le=10, description="Chunks after reranking")
    use_reranking: Optional[bool] = Field(None, description="Enable/disable cross-encoder reranking")

    @field_validator("question")
    @classmethod
    def strip_question(cls, v: str) -> str:
        return v.strip()


class SwitchModelRequest(BaseModel):
    model: str = Field(..., description="Model name to switch to")

    @field_validator("model")
    @classmethod
    def validate_model(cls, v: str) -> str:
        import config
        if v not in config.LLM_MODELS:
            raise ValueError(
                f"Unknown model '{v}'. Available: {list(config.LLM_MODELS)}"
            )
        return v


# ─── Response schemas ─────────────────────────────────────────────────────────


class SourceItem(BaseModel):
    source: str
    page_number: int
    score: float
    excerpt: str


class ChatResponse(BaseModel):
    question: str
    answer: str
    sources: List[SourceItem]
    model: str
    retrieved_count: int
    usage: Dict[str, Any] = Field(default_factory=dict)


class UploadResponse(BaseModel):
    filename: str
    pages_processed: int
    chunks_created: int
    message: str


class HealthResponse(BaseModel):
    status: str
    version: str
    active_model: str
    active_embedding: str
    vector_store_backend: str
    document_count: int
    gpu_info: Dict[str, Any] = Field(default_factory=dict)


class ModelInfo(BaseModel):
    name: str
    provider: str
    description: str
    loaded: bool
    active: bool
    max_tokens: int
    context_window: int


class ModelsResponse(BaseModel):
    active_model: str
    models: List[ModelInfo]


class SwitchModelResponse(BaseModel):
    previous_model: str
    current_model: str
    message: str


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
