"""
Automated tests for DocuBot RAG + agentic validation (no live Gemini calls).

LLM behavior is simulated with ``unittest.mock`` on the client's public methods.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from docubot import DocuBot
from pipeline_models import UserQuery, ValidationResult
from rag_logger import RagInteractionLogger
from rag_pipeline import REFUSAL_TEXT, run_rag_pipeline


@pytest.fixture
def log_to_tmp(tmp_path):
    """Avoid writing under logs/ during tests."""
    return RagInteractionLogger(log_path=tmp_path / "rag_test.jsonl")


@pytest.fixture
def mock_llm():
    """Stand-in for GeminiClient; no network or API key required."""
    m = MagicMock()
    return m


@pytest.fixture
def bot_with_llm(mock_llm, tmp_path):
    """
    Real DocuBot index (loads project docs/) with a mocked LLM client.

    Retrieval is patched per test so scoring stays deterministic.
    """
    # Minimal docs tree so DocuBot.__init__ is fast even if cwd varies.
    ddir = tmp_path / "docs"
    ddir.mkdir()
    (ddir / "stub.md").write_text("Stub documentation for tests.\n", encoding="utf-8")
    return DocuBot(docs_folder=str(ddir), llm_client=mock_llm)


def test_golden_path_initial_draft_passes_validation(
    bot_with_llm, mock_llm, log_to_tmp, monkeypatch
):
    """Accurate draft + high-confidence pass → user sees the draft; no retry."""
    snippets = [
        ("API_REFERENCE.md", "List users returns a paginated JSON array. Max page size is 50."),
    ]
    grounded_answer = (
        "Per API_REFERENCE.md, list users returns a paginated JSON array; max page size is 50."
    )

    monkeypatch.setattr(bot_with_llm, "retrieve", lambda q, top_k=3: list(snippets))
    mock_llm.answer_from_snippets.return_value = grounded_answer
    mock_llm.validate_rag_draft.return_value = ValidationResult(
        passed=True,
        confidence_score=0.94,
        reasoning="Claims match the cited snippet.",
    )

    out = run_rag_pipeline(
        bot_with_llm,
        UserQuery(text="How does user listing work?"),
        logger=log_to_tmp,
    )

    assert out == grounded_answer
    mock_llm.answer_from_snippets.assert_called_once()
    mock_llm.validate_rag_draft.assert_called_once()
    mock_llm.answer_from_snippets_retry.assert_not_called()


def test_hallucination_caught_then_refusal_after_retry(
    bot_with_llm, mock_llm, log_to_tmp, monkeypatch
):
    """
    First draft invents facts; validator fails (simulated). Retry does not recover;
    pipeline returns the safe refusal string.
    """
    snippets = [
        ("AUTH.md", "Access tokens expire after one hour. Refresh using the refresh endpoint."),
    ]
    hallucinated_draft = (
        "Per AUTH.md, access tokens expire after one hour. "
        "You must also call SuperSecretEndpoint999 to activate quantum auth."
    )
    retry_draft = "Still wrong: undocumented MegaTableX is required for login."

    monkeypatch.setattr(bot_with_llm, "retrieve", lambda q, top_k=3: list(snippets))
    mock_llm.answer_from_snippets.return_value = hallucinated_draft
    mock_llm.answer_from_snippets_retry.return_value = retry_draft
    mock_llm.validate_rag_draft.side_effect = [
        ValidationResult(
            passed=False,
            confidence_score=0.12,
            reasoning="SuperSecretEndpoint999 is not supported by the snippets.",
        ),
        ValidationResult(
            passed=False,
            confidence_score=0.18,
            reasoning="MegaTableX is not mentioned in AUTH.md.",
        ),
    ]

    out = run_rag_pipeline(
        bot_with_llm,
        UserQuery(text="How do tokens work?"),
        logger=log_to_tmp,
    )

    assert out == REFUSAL_TEXT
    mock_llm.answer_from_snippets.assert_called_once()
    mock_llm.answer_from_snippets_retry.assert_called_once()
    assert mock_llm.validate_rag_draft.call_count == 2

    first_validate_args = mock_llm.validate_rag_draft.call_args_list[0][0]
    assert first_validate_args[1] == hallucinated_draft
    assert "SuperSecretEndpoint999" in first_validate_args[1]


def test_empty_context_no_generation_or_validation(
    bot_with_llm, mock_llm, log_to_tmp, monkeypatch
):
    """No snippets → immediate refusal; no draft or validator LLM calls."""
    monkeypatch.setattr(bot_with_llm, "retrieve", lambda q, top_k=3: [])

    out = run_rag_pipeline(
        bot_with_llm,
        UserQuery(text="something that matches nothing in the index"),
        logger=log_to_tmp,
    )

    assert out == REFUSAL_TEXT
    mock_llm.answer_from_snippets.assert_not_called()
    mock_llm.answer_from_snippets_retry.assert_not_called()
    mock_llm.validate_rag_draft.assert_not_called()
