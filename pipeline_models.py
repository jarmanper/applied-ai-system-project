"""
Structured types for the DocuBot RAG pipeline (retrieval → draft → validation).

Validation fields are populated by a future LLM validator; the live pipeline
currently uses a placeholder that always passes.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class UserQuery(BaseModel):
    """End-user question entering the pipeline."""

    text: str = Field(..., min_length=1, description="Natural language question")
    top_k: int = Field(default=3, ge=1, le=20, description="Max snippets to retrieve")


class ContextSnippet(BaseModel):
    """One retrieved evidence block."""

    filename: str
    text: str


class RetrievedContext(BaseModel):
    """All snippets passed into draft generation."""

    snippets: list[ContextSnippet] = Field(default_factory=list)


class DraftAnswer(BaseModel):
    """Initial model output before validation / revision."""

    text: str
    model_name: str | None = None


class ValidationResult(BaseModel):
    """Outcome of the answer-quality check (LLM-backed in a later stage)."""

    passed: bool
    confidence_score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Normalized confidence when the validator supplies it",
    )
    reasoning: str | None = Field(
        default=None, description="Short validator rationale when available"
    )
