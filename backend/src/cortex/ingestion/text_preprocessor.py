"""
Central Text Preprocessor.

Implements §6.8 of the Canonical Blueprint.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, Literal, Optional, Protocol, Tuple

from cortex.email_processing import clean_email_text
from cortex.ingestion.pii import redact_pii

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
        self.cleaning_version = "v1"
        # Control chars: keep tab, newline, carriage return
        self._control_chars = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]")

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
        cleaned = self._strip_control_chars(text)
        
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
        meta.update({
            "pre_cleaned": True,
            "cleaning_version": self.cleaning_version,
            "source": text_type,
            "char_count_raw": len(text),
            "char_count_clean": len(cleaned),
        })
        
        return cleaned, meta

    def _strip_control_chars(self, text: str) -> str:
        """Remove non-printable control characters."""
        return self._control_chars.sub("", text)

    def _redact_pii(self, text: str) -> str:
        """
        Redact PII from text using the centralized PII engine.
        """
        return redact_pii(text)


_preprocessor: Optional[TextPreprocessor] = None


def get_text_preprocessor() -> TextPreprocessor:
    """Get the global TextPreprocessor instance."""
    global _preprocessor
    if _preprocessor is None:
        _preprocessor = DefaultTextPreprocessor()
    return _preprocessor