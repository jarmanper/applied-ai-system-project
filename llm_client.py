"""
Gemini client wrapper used by DocuBot.

Handles:
- Configuring the Gemini client from the GEMINI_API_KEY environment variable
- Naive "generation only" answers over the full docs corpus (Phase 0)
- RAG style answers that use only retrieved snippets (Phase 2)
- Structured self-critique / validation of RAG drafts (Phase 2b)

Experiment with:
- Prompt wording
- Refusal conditions
- How strictly the model is instructed to use only the provided context
"""

from __future__ import annotations

import json
import os
import re

import google.generativeai as genai

from pipeline_models import ValidationResult

# Central place to update the model name if needed.
# You can swap this for a different Gemini model in the future.
GEMINI_MODEL_NAME = "gemini-2.5-flash"

# Minimum validator confidence (0–1) required after the model's own pass/fail.
_DEFAULT_MIN_CONFIDENCE = 0.72
RAG_VALIDATION_MIN_CONFIDENCE = float(
    os.getenv("RAG_VALIDATION_MIN_CONFIDENCE", str(_DEFAULT_MIN_CONFIDENCE))
)

_VALIDATOR_JSON_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "passed": {"type": "boolean"},
        "confidence_score": {"type": "number"},
        "reasoning": {"type": "string"},
    },
    "required": ["passed", "confidence_score", "reasoning"],
}

_VALIDATOR_GENERATION_CONFIG = genai.GenerationConfig(
    response_mime_type="application/json",
    response_schema=_VALIDATOR_JSON_SCHEMA,
    temperature=0.15,
    max_output_tokens=1024,
)


def _extract_text(response) -> str:
    """Best-effort plain text from a generate_content response."""
    t = getattr(response, "text", None) or ""
    if t:
        return t
    try:
        parts = response.candidates[0].content.parts
        return "".join(getattr(p, "text", "") or "" for p in parts)
    except (AttributeError, IndexError, KeyError):
        return ""


def _format_snippet_blocks(snippets: list[tuple[str, str]]) -> str:
    blocks = []
    for filename, text in snippets:
        blocks.append(f"File: {filename}\n{text}\n")
    return "\n\n".join(blocks)


def _parse_validator_json(raw: str) -> ValidationResult:
    raw = raw.strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]*\}\s*$", raw)
        if not m:
            raise
        data = json.loads(m.group(0))
    return ValidationResult.model_validate(data)


def _apply_confidence_guardrail(
    result: ValidationResult, min_confidence: float
) -> ValidationResult:
    """Require minimum confidence even if the model sets passed=true."""
    s = result.confidence_score
    if s is None:
        s = 0.0
    s = max(0.0, min(1.0, float(s)))
    ok = bool(result.passed) and s >= min_confidence
    reason = result.reasoning or ""
    if bool(result.passed) and not ok:
        reason = (
            reason
            + f" [system guardrail: confidence {s:.2f} < required {min_confidence:.2f}]"
        ).strip()
    return ValidationResult(passed=ok, confidence_score=s, reasoning=reason or None)


class GeminiClient:
    """
    Simple wrapper around the Gemini model.

    Usage:
        client = GeminiClient()
        answer = client.naive_answer_over_full_docs(query, all_text)
        # or
        answer = client.answer_from_snippets(query, snippets)
    """

    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Missing GEMINI_API_KEY environment variable. "
                "Set it in your shell or .env file to enable LLM features."
            )

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(GEMINI_MODEL_NAME)

    # -----------------------------------------------------------
    # Phase 0: naive generation over full docs
    # -----------------------------------------------------------

    def naive_answer_over_full_docs(self, query, all_text):
        prompt = f"""
You are a documentation assistant. Your ONLY allowed facts are in the documentation below.

Documentation:
{all_text}

Answer the developer question using only that documentation. If it does not contain
enough information, reply exactly:
"I do not know based on these docs."

Developer question:
{query}
"""
        response = self.model.generate_content(prompt)
        return _extract_text(response).strip()

    # -----------------------------------------------------------
    # Phase 2: RAG style generation over retrieved snippets
    # -----------------------------------------------------------

    def answer_from_snippets(self, query, snippets):
        """
        Phase 2:
        Generate an answer using only the retrieved snippets.

        snippets: list of (filename, text) tuples selected by DocuBot.retrieve

        The prompt:
        - Shows each snippet with its filename
        - Instructs the model to rely only on these snippets
        - Requires an explicit "I do not know" refusal when needed
        """

        if not snippets:
            return "I do not know based on these docs."

        context = _format_snippet_blocks(snippets)

        prompt = f"""
You are a cautious documentation assistant helping developers understand a codebase.

You will receive:
- A developer question
- A small set of snippets from project files

Your job:
- Answer the question using only the information in the snippets.
- If the snippets do not provide enough evidence, refuse to guess.

Snippets:
{context}

Developer question:
{query}

Rules:
- Use only the information in the snippets. Do not invent new functions,
  endpoints, or configuration values.
- If the snippets are not enough to answer confidently, reply exactly:
  "I do not know based on these docs."
- When you do answer, briefly mention which files you relied on.
"""

        response = self.model.generate_content(prompt)
        return _extract_text(response).strip()

    def answer_from_snippets_retry(
        self,
        query: str,
        snippets: list[tuple[str, str]],
        failed_draft: str,
        validation_reasoning: str | None,
    ) -> str:
        """
        Second-pass draft after a failed validation: same evidence, stricter
        self-correction using the validator's rationale.
        """
        if not snippets:
            return "I do not know based on these docs."

        context = _format_snippet_blocks(snippets)
        feedback = (validation_reasoning or "").strip() or "No detailed rationale provided."

        prompt = f"""
You are a cautious documentation assistant. Your FIRST draft failed an internal
quality check. Produce a NEW answer using ONLY the snippets below.

Snippets (sole source of truth):
{context}

Original developer question:
{query}

Failed draft (do NOT repeat its mistakes):
{failed_draft}

Internal review notes (address these issues; do not contradict the snippets):
{feedback}

Rules:
- Use only information supported by the snippets. If they are still insufficient,
  reply exactly: "I do not know based on these docs."
- Do not invent APIs, tables, env vars, or behaviors not evidenced in the snippets.
- When you answer, briefly cite which files you used.
"""

        response = self.model.generate_content(prompt)
        return _extract_text(response).strip()

    def validate_rag_draft(
        self,
        query: str,
        draft_text: str,
        snippets: list[tuple[str, str]],
        *,
        min_confidence: float | None = None,
    ) -> ValidationResult:
        """
        Secondary Gemini call: judge whether the draft is grounded in the snippets.

        Returns a ValidationResult after JSON parsing and system guardrails.
        """
        threshold = (
            float(min_confidence)
            if min_confidence is not None
            else RAG_VALIDATION_MIN_CONFIDENCE
        )
        context = _format_snippet_blocks(snippets)

        prompt = f"""
You are an independent fact-checker for a retrieval-augmented documentation bot.

You will receive:
1) The same retrieved file snippets the author saw (these snippets are the ONLY
   acceptable evidence of what the codebase docs say).
2) The user's question.
3) A draft answer that claims to be based on those snippets.

Your task:
- Decide if the draft is FULLY supported by the snippets (no contradictions,
  no invented facts, no overreach beyond what the text actually states).
- Detect hallucinations: names, flags, endpoints, SQL, behaviors, or config keys
  that do not appear or clearly follow from the snippets.
- If the draft hedges correctly with "I do not know based on these docs." when
  evidence is weak, that is acceptable when the snippets truly lack support.

Scoring:
- confidence_score: a number from 0.0 (no support / unreliable) to 1.0 (strongly
  supported and complete relative to the question and snippets).
- passed: true ONLY if the draft is well grounded AND confidence_score >= {threshold:.2f}.
  If you have any material doubt, set passed to false and keep confidence low.

Respond ONLY with a JSON object (no markdown) matching the required schema fields:
passed, confidence_score, reasoning (reasoning: one short paragraph).

User question:
{query}

Snippets:
{context}

Draft answer to evaluate:
{draft_text}
"""

        try:
            response = self.model.generate_content(
                prompt, generation_config=_VALIDATOR_GENERATION_CONFIG
            )
        except Exception as exc:
            return ValidationResult(
                passed=False,
                confidence_score=0.0,
                reasoning=f"Validator API error: {type(exc).__name__}: {exc}",
            )

        raw = _extract_text(response).strip()
        if not raw:
            return ValidationResult(
                passed=False,
                confidence_score=0.0,
                reasoning="Validator returned empty output.",
            )
        try:
            parsed = _parse_validator_json(raw)
        except Exception as exc:
            return ValidationResult(
                passed=False,
                confidence_score=0.0,
                reasoning=f"Validator JSON/schema error: {type(exc).__name__}: {exc}",
            )
        return _apply_confidence_guardrail(parsed, threshold)
