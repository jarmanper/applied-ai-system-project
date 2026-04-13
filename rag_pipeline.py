"""
RAG orchestration: retrieval → draft generation → LLM validation (with one retry).

If validation still fails after a single regenerated draft, the user sees the
standard refusal string.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from llm_client import GEMINI_MODEL_NAME, RAG_VALIDATION_MIN_CONFIDENCE
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

REFUSAL_TEXT = "I do not know based on these docs."


def _snippet_tuples(ctx: RetrievedContext) -> list[tuple[str, str]]:
    return [(s.filename, s.text) for s in ctx.snippets]


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
    tuples = _snippet_tuples(ctx)
    text = bot.llm_client.answer_from_snippets(user_query.text, tuples)
    return DraftAnswer(text=text, model_name=GEMINI_MODEL_NAME)


def draft_generation_retry(
    bot: DocuBot,
    user_query: UserQuery,
    ctx: RetrievedContext,
    failed_draft: DraftAnswer,
    validation: ValidationResult,
) -> DraftAnswer:
    """Single retry after a failed validation, conditioned on validator reasoning."""
    if bot.llm_client is None:
        raise RuntimeError(
            "RAG mode requires an LLM client. Provide a GeminiClient instance."
        )
    tuples = _snippet_tuples(ctx)
    text = bot.llm_client.answer_from_snippets_retry(
        user_query.text,
        tuples,
        failed_draft.text,
        validation.reasoning,
    )
    return DraftAnswer(text=text, model_name=GEMINI_MODEL_NAME)


def validate_draft_with_llm(
    bot: DocuBot,
    user_query: UserQuery,
    draft: DraftAnswer,
    ctx: RetrievedContext,
) -> ValidationResult:
    """Secondary Gemini call: groundedness and confidence vs snippets."""
    if bot.llm_client is None:
        raise RuntimeError(
            "RAG mode requires an LLM client. Provide a GeminiClient instance."
        )
    return bot.llm_client.validate_rag_draft(
        user_query.text,
        draft.text,
        _snippet_tuples(ctx),
    )


def run_rag_pipeline(
    bot: DocuBot,
    user_query: UserQuery,
    *,
    logger: RagInteractionLogger | None = None,
) -> str:
    """
    End-to-end RAG path: retrieve → draft → validate → (optional one retry) → answer.

    The user only sees a draft after it passes validation; otherwise they see
    ``REFUSAL_TEXT`` after at most two draft attempts.
    """
    log = logger or get_default_rag_logger()
    retrieved: RetrievedContext | None = None
    draft: DraftAnswer | None = None
    validation: ValidationResult | None = None
    rag_attempts: list[dict] = []
    final_answer: str | None = None
    used_retry = False

    try:
        retrieved = retrieve_context(bot, user_query)

        if not retrieved.snippets:
            final_answer = REFUSAL_TEXT
            log.log_pipeline_record(
                {
                    "event": "rag_pipeline",
                    "validation_min_confidence": RAG_VALIDATION_MIN_CONFIDENCE,
                    "user_query": user_query.model_dump(),
                    "retrieved_context": retrieved.model_dump(),
                    "draft_answer": None,
                    "validation_result": None,
                    "rag_attempts": [],
                    "used_retry": False,
                    "final_answer": final_answer,
                    "error": None,
                }
            )
            return final_answer

        draft = draft_generation(bot, user_query, retrieved)
        validation = validate_draft_with_llm(bot, user_query, draft, retrieved)
        rag_attempts.append(
            {
                "attempt": 1,
                "draft": draft.model_dump(),
                "validation": validation.model_dump(),
            }
        )

        if validation.passed:
            final_answer = draft.text
        else:
            used_retry = True
            draft = draft_generation_retry(
                bot, user_query, retrieved, draft, validation
            )
            validation = validate_draft_with_llm(bot, user_query, draft, retrieved)
            rag_attempts.append(
                {
                    "attempt": 2,
                    "draft": draft.model_dump(),
                    "validation": validation.model_dump(),
                }
            )
            final_answer = draft.text if validation.passed else REFUSAL_TEXT

        log.log_pipeline_record(
            {
                "event": "rag_pipeline",
                "validation_min_confidence": RAG_VALIDATION_MIN_CONFIDENCE,
                "user_query": user_query.model_dump(),
                "retrieved_context": retrieved.model_dump(),
                "draft_answer": draft.model_dump() if draft else None,
                "validation_result": validation.model_dump() if validation else None,
                "rag_attempts": rag_attempts,
                "used_retry": used_retry,
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
                "validation_min_confidence": RAG_VALIDATION_MIN_CONFIDENCE,
                "user_query": user_query.model_dump(),
                "retrieved_context": retrieved.model_dump() if retrieved else None,
                "draft_answer": draft.model_dump() if draft else None,
                "validation_result": validation.model_dump() if validation else None,
                "rag_attempts": rag_attempts,
                "used_retry": used_retry,
                "final_answer": final_answer,
                "error": error_message,
            }
        )
        raise
