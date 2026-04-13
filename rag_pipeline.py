"""
RAG orchestration: retrieval → draft generation → validation.

LLM-based validation is not implemented yet; ``validate_draft_placeholder``
always passes so later stages can swap in a real validator without changing
the outer flow.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from llm_client import GEMINI_MODEL_NAME
from pipeline_models import (
    ContextSnippet,
    DraftAnswer,
    RetrievedContext,
    UserQuery,
    ValidationResult,
)
from rag_logger import RagInteractionLogger, get_default_rag_logger

if TYPE_CHECKING:
    from docubot import DocuBot


def retrieve_context(bot: DocuBot, user_query: UserQuery) -> RetrievedContext:
    """Select evidence snippets for the query."""
    rows = bot.retrieve(user_query.text, top_k=user_query.top_k)
    return RetrievedContext(
        snippets=[ContextSnippet(filename=fname, text=text) for fname, text in rows]
    )


def draft_generation(bot: DocuBot, user_query: UserQuery, ctx: RetrievedContext) -> DraftAnswer:
    """Call Gemini with retrieved snippets only (initial answer)."""
    if bot.llm_client is None:
        raise RuntimeError(
            "RAG mode requires an LLM client. Provide a GeminiClient instance."
        )
    tuples = [(s.filename, s.text) for s in ctx.snippets]
    text = bot.llm_client.answer_from_snippets(user_query.text, tuples)
    return DraftAnswer(text=text, model_name=GEMINI_MODEL_NAME)


def validate_draft_placeholder(
    draft: DraftAnswer,
    user_query: UserQuery,
    ctx: RetrievedContext,
) -> ValidationResult:
    """
    Reserved hook for an LLM validation loop.

    Currently returns a trivial pass so downstream code can branch on
    ``ValidationResult.passed`` when the real validator lands.
    """
    _ = (draft, user_query, ctx)
    return ValidationResult(
        passed=True,
        confidence_score=None,
        reasoning="Placeholder: validator not wired; unconditionally passes.",
    )


def run_rag_pipeline(
    bot: DocuBot,
    user_query: UserQuery,
    *,
    logger: RagInteractionLogger | None = None,
) -> str:
    """
    End-to-end RAG path with explicit draft and validation stages.

    Returns the user-facing answer string (today: the draft when validation passes).
    """
    log = logger or get_default_rag_logger()
    retrieved: RetrievedContext | None = None
    draft: DraftAnswer | None = None
    validation: ValidationResult | None = None
    final_answer: str | None = None
    error_message: str | None = None

    try:
        retrieved = retrieve_context(bot, user_query)

        if not retrieved.snippets:
            final_answer = "I do not know based on these docs."
            log.log_pipeline_record(
                {
                    "event": "rag_pipeline",
                    "user_query": user_query.model_dump(),
                    "retrieved_context": retrieved.model_dump(),
                    "draft_answer": None,
                    "validation_result": None,
                    "final_answer": final_answer,
                    "error": None,
                }
            )
            return final_answer

        draft = draft_generation(bot, user_query, retrieved)
        validation = validate_draft_placeholder(draft, user_query, retrieved)
        # Placeholder always passes; a future validator will choose the surfaced answer.
        final_answer = draft.text

        log.log_pipeline_record(
            {
                "event": "rag_pipeline",
                "user_query": user_query.model_dump(),
                "retrieved_context": retrieved.model_dump(),
                "draft_answer": draft.model_dump(),
                "validation_result": validation.model_dump(),
                "final_answer": final_answer,
                "error": None,
            }
        )
        return final_answer

    except Exception as exc:
        error_message = f"{type(exc).__name__}: {exc}"
        log.log_pipeline_record(
            {
                "event": "rag_pipeline",
                "user_query": user_query.model_dump(),
                "retrieved_context": retrieved.model_dump() if retrieved else None,
                "draft_answer": draft.model_dump() if draft else None,
                "validation_result": validation.model_dump() if validation else None,
                "final_answer": final_answer,
                "error": error_message,
            }
        )
        raise
