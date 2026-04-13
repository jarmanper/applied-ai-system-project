"""
Append-only JSONL logging for RAG pipeline runs.

Designed for auditability without persisting secrets: optional redaction and
size limits keep log lines bounded and safer to share.
"""

from __future__ import annotations

import json
import os
import re
import threading
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Typical Google API key prefix; used only for defensive redaction in free text.
_API_KEY_PATTERN = re.compile(r"AIza[0-9A-Za-z\-_]{20,}", re.ASCII)

_DEFAULT_LOG_PATH = Path("logs") / "rag_interactions.jsonl"
_DEFAULT_MAX_VALUE_CHARS = 16_384


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def redact_secrets(obj: Any, max_chars: int = _DEFAULT_MAX_VALUE_CHARS) -> Any:
    """
    Recursively copy structures for logging: truncate long strings and scrub
    obvious API key material from values.
    """
    if isinstance(obj, str):
        s = _API_KEY_PATTERN.sub("[REDACTED_API_KEY]", obj)
        if len(s) > max_chars:
            s = s[: max_chars - 20] + "…[truncated]"
        return s
    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for k, v in obj.items():
            lk = str(k).lower()
            if any(
                token in lk
                for token in ("api_key", "apikey", "password", "secret", "token", "authorization")
            ):
                out[str(k)] = "[REDACTED]"
            else:
                out[str(k)] = redact_secrets(v, max_chars=max_chars)
        return out
    if isinstance(obj, list):
        return [redact_secrets(x, max_chars=max_chars) for x in obj]
    if isinstance(obj, tuple):
        return [redact_secrets(x, max_chars=max_chars) for x in obj]
    return obj


class RagInteractionLogger:
    """
    Thread-safe JSONL writer under logs/ (directory is created on demand).
    """

    def __init__(
        self,
        log_path: os.PathLike[str] | str | None = None,
        max_value_chars: int = _DEFAULT_MAX_VALUE_CHARS,
    ) -> None:
        self._path = Path(log_path) if log_path is not None else _DEFAULT_LOG_PATH
        self._max_value_chars = max_value_chars
        self._lock = threading.Lock()

    @property
    def log_path(self) -> Path:
        return self._path

    def log_pipeline_record(self, record: dict[str, Any]) -> None:
        """
        Append one JSON object per line. ``record`` should be JSON-serializable
        after redaction (plain dicts, lists, strings, numbers, booleans, None).
        """
        payload = {
            "timestamp": _utc_now_iso(),
            **record,
        }
        safe = redact_secrets(payload, max_chars=self._max_value_chars)
        line = json.dumps(safe, ensure_ascii=True) + "\n"
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            with self._path.open("a", encoding="utf-8") as f:
                f.write(line)

    def log_exception(self, *, stage: str, error: BaseException, context: dict[str, Any] | None = None) -> None:
        """Record a structured error without re-raising."""
        rec: dict[str, Any] = {
            "event": "error",
            "stage": stage,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "traceback": traceback.format_exc(),
        }
        if context:
            rec["context"] = context
        self.log_pipeline_record(rec)


_default_logger: RagInteractionLogger | None = None
_default_lock = threading.Lock()


def get_default_rag_logger() -> RagInteractionLogger:
    global _default_logger
    with _default_lock:
        if _default_logger is None:
            _default_logger = RagInteractionLogger()
        return _default_logger
