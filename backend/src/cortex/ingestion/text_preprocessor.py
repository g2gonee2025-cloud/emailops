"""
Central Text Preprocessor.

Implements §6.8 of the Canonical Blueprint.
"""

from __future__ import annotations

import logging
import re
import threading
from typing import Any, Dict, Literal, Optional, Protocol, Tuple

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
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, Dict[str, Any]]:
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
    1. Type-specific cleaning (email headers, signatures)
    2. Whitespace normalization
    3. Control character stripping
    4. PII redaction (placeholder)
    """

    def __init__(self):
        self.cleaning_version = CLEANING_VERSION

    def prepare_for_indexing(
        self,
        text: str,
        *,
        text_type: Literal["email", "attachment"],
        tenant_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Clean and prepare text for indexing.
        """
        meta = (metadata or {}).copy()

        # 1. Basic sanitization
        cleaned = strip_control_chars(text)

        # 2. Type-specific cleaning
        if text_type == "email":
            # Use existing email cleaning logic
            cleaned = clean_email_text(cleaned)
        elif text_type == "attachment":
            # Basic whitespace normalization for docs
            cleaned = re.sub(r"\s+", " ", cleaned).strip()

        # 3. PII Redaction (§6.4)
        cleaned = self._redact_pii(cleaned)

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
        config = get_config()
        if not config.pii.enabled:
            return text

        return redact_pii(text)


_preprocessor: Optional[TextPreprocessor] = None


_preprocessor_lock = threading.Lock()


def get_text_preprocessor() -> TextPreprocessor:
    """Get the global TextPreprocessor instance."""
    global _preprocessor
    if _preprocessor is None:
        with _preprocessor_lock:
            if _preprocessor is None:
                _preprocessor = DefaultTextPreprocessor()
    return _preprocessor
