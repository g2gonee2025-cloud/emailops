"""
Central Text Preprocessor.

Implements §6.8 of the Canonical Blueprint.
"""

from __future__ import annotations

import logging
import re
import threading
from collections.abc import Mapping
from typing import Any, Literal, Protocol

from cortex.config.loader import get_config
from cortex.email_processing import clean_email_text
from cortex.ingestion.constants import CLEANING_VERSION
from cortex.ingestion.pii import redact_pii
from cortex.ingestion.text_utils import strip_control_chars

logger = logging.getLogger(__name__)


class TextPreprocessor(Protocol):
    """Protocol for text preprocessing."""

    def prepare_for_indexing(
        self,
        text: str,
        *,
        text_type: Literal["email", "attachment"],
        tenant_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """
        Prepare text for indexing by cleaning and redacting PII.

        Args:
            text: Raw text content
            text_type: Source type ("email" or "attachment")
            tenant_id: Tenant identifier
            metadata: Optional metadata to augment

        Returns:
            Tuple of (cleaned_text, updated_metadata)
        """
        ...


class DefaultTextPreprocessor:
    """
    Default implementation of TextPreprocessor.

    Applies:
    1. Control character stripping
    2. Type-specific cleaning (email headers, signatures)
    3. Whitespace normalization (attachments only)
    4. PII redaction
    """

    def __init__(self):
        self.cleaning_version = CLEANING_VERSION
        self._pii_enabled: bool | None = None

    def prepare_for_indexing(
        self,
        text: str,
        *,
        text_type: Literal["email", "attachment"],
        tenant_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """
        Clean and prepare text for indexing.
        """
        meta = dict(metadata) if isinstance(metadata, Mapping) else {}
        if metadata is not None and not isinstance(metadata, Mapping):
            logger.warning(
                "Invalid metadata type (%s); using empty metadata", type(metadata)
            )

        if not isinstance(text, str):
            logger.warning("Non-string text received; coercing to string")
            text = "" if text is None else str(text)

        if text_type not in {"email", "attachment"}:
            meta["text_type_original"] = str(text_type)
            logger.warning(
                "Unexpected text_type %s; defaulting to attachment cleaning",
                text_type,
            )
            text_type = "attachment"

        # 1. Basic sanitization
        try:
            cleaned = strip_control_chars(text)
        except Exception:
            logger.exception("Failed to strip control chars; using raw text")
            cleaned = text

        # 2. Type-specific cleaning
        if text_type == "email":
            # Use existing email cleaning logic
            try:
                cleaned = clean_email_text(cleaned)
            except Exception:
                logger.exception("Failed to clean email text; using sanitized text")
        elif text_type == "attachment":
            # Basic whitespace normalization for docs
            try:
                cleaned = re.sub(r"\s+", " ", cleaned).strip()
            except Exception:
                logger.exception(
                    "Failed to normalize attachment whitespace; using sanitized text"
                )

        # 3. PII Redaction (§6.4)
        try:
            cleaned = self._redact_pii(cleaned)
        except Exception:
            logger.exception("PII redaction failed; using cleaned text")

        # 4. Update metadata
        meta.update(
            {
                "pre_cleaned": True,
                "cleaning_version": self.cleaning_version,
                "source": text_type,
                "char_count_raw": len(text),
                "char_count_clean": len(cleaned),
            }
        )

        return cleaned, meta

    def _redact_pii(self, text: str) -> str:
        """
        Redact PII from text using the centralized PII engine.
        """
        if self._pii_enabled is None:
            self._pii_enabled = _get_pii_enabled()
        if not self._pii_enabled:
            return text

        return redact_pii(text)


def _get_pii_enabled() -> bool:
    config = get_config()
    pii_config = getattr(config, "pii", None)
    if isinstance(config, Mapping):
        pii_config = config.get("pii")
    if isinstance(pii_config, Mapping):
        return bool(pii_config.get("enabled"))
    enabled = getattr(pii_config, "enabled", None)
    return bool(enabled)


_preprocessor: TextPreprocessor | None = None


_preprocessor_lock = threading.Lock()


def get_text_preprocessor() -> TextPreprocessor:
    """Get the global TextPreprocessor instance."""
    global _preprocessor
    if _preprocessor is None:
        with _preprocessor_lock:
            if _preprocessor is None:
                _preprocessor = DefaultTextPreprocessor()
    return _preprocessor
