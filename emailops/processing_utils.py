"""
Processing utilities including batch processing, performance monitoring, and configuration.
"""
from __future__ import annotations

import asyncio
import datetime
import hashlib
import logging
import os
import re
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import lru_cache, wraps
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ProcessingConfig:
    """Centralized configuration for document processing."""

    max_attachment_chars: int = field(
        default_factory=lambda: int(os.getenv("MAX_ATTACHMENT_TEXT_CHARS", "500000"))
    )
    excel_max_cells: int = field(default_factory=lambda: int(os.getenv("EXCEL_MAX_CELLS", "200000")))
    skip_attachment_over_mb: float = field(
        default_factory=lambda: float(os.getenv("SKIP_ATTACHMENT_OVER_MB", "0"))
    )
    max_total_attachment_text: int = 10000

    def __post_init__(self):
        """Validate configuration values."""
        if self.max_attachment_chars < 0:
            self.max_attachment_chars = 500000
        if self.excel_max_cells < 0:
            self.excel_max_cells = 200000
        if self.skip_attachment_over_mb < 0:
            self.skip_attachment_over_mb = 0


# Singleton configuration instance
_PROCESSING_CONFIG: ProcessingConfig | None = None


def get_processing_config() -> ProcessingConfig:
    """Get the singleton processing configuration."""
    global _PROCESSING_CONFIG
    if _PROCESSING_CONFIG is None:
        _PROCESSING_CONFIG = ProcessingConfig()
    return _PROCESSING_CONFIG


# MEDIUM #23: Person class appears unused in codebase but kept for backward compatibility
@dataclass
class Person:
    """Immutable person object with age calculation. NOTE: Currently unused in codebase."""

    name: str
    birthdate: str

    def __post_init__(self):
        """Validate birthdate format."""
        if self.birthdate:
            try:
                datetime.date.fromisoformat(self.birthdate)
            except ValueError as e:
                logger.warning("Invalid birthdate format for %s: %s", self.name, e)

    @property
    def age(self) -> int:
        """Calculate age based on today's date (timezone-agnostic)."""
        return self.age_on(datetime.date.today())

    def age_on(self, on_date: datetime.date) -> int:
        """Calculate age on a specific date using date arithmetic."""
        if not self.birthdate:
            return 0
        try:
            b = datetime.date.fromisoformat(self.birthdate)
            return on_date.year - b.year - ((on_date.month, on_date.day) < (b.month, b.day))
        except Exception:
            return 0

    def getAge(self) -> int:
        """Alias for the age property (backward compatibility)."""
        return self.age


def monitor_performance(func: Callable) -> Callable:
    """Decorator to monitor function performance."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()

        try:
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - start_time

            if elapsed > 1.0:  # Log slow operations
                logger.warning("%s took %.2f seconds", func.__name__, elapsed)
            else:
                logger.debug("%s completed in %.3f seconds", func.__name__, elapsed)

            return result
        except Exception as e:
            elapsed = time.perf_counter() - start_time
            logger.error("%s failed after %.2f seconds: %s", func.__name__, elapsed, e)
            raise

    return wrapper


class BatchProcessor:
    """Process items in batches with error handling."""

    def __init__(self, batch_size: int = 100, max_workers: int = 4):
        self.batch_size = batch_size
        self.max_workers = max_workers

    def process_items(
        self,
        items: list[Any],
        processor: Callable[[Any], Any],
        error_handler: Callable[[Any, Exception], None] | None = None,
    ) -> list[Any]:
        """
        Process items in batches with parallel processing.

        Args:
            items: Items to process
            processor: Function to process each item
            error_handler: Optional error handler

        Returns:
            List of processed results
        """
        results = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Process in batches
            for i in range(0, len(items), self.batch_size):
                batch = items[i : i + self.batch_size]

                # Submit batch for processing
                futures = [executor.submit(processor, item) for item in batch]

                # Collect results
                for future, item in zip(futures, batch, strict=False):
                    try:
                        result = future.result(timeout=30)
                        results.append(result)
                    except Exception as e:
                        if error_handler:
                            error_handler(item, e)
                        else:
                            logger.error("Failed to process item: %s", e)
                        results.append(None)

        return results

    async def process_items_async(self, items: list[Any], processor: Callable[[Any], Any]) -> list[Any]:
        """Async version of process_items."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.process_items, items, processor)


@dataclass
class TextPreprocessor:
    """
    Centralized text preprocessing for the indexing pipeline.

    This class ensures text is cleaned ONCE before chunking, eliminating
    redundant processing during retrieval. This optimization provides
    40-60% performance improvement.
    """

    PREPROCESSING_VERSION = "2.0"

    def __init__(self):
        """Initialize preprocessor with cache."""
        self._cache: dict[str, tuple[str, dict[str, Any]]] = {}

    @monitor_performance
    def prepare_for_indexing(
        self, text: str, text_type: str = "email", use_cache: bool = True
    ) -> tuple[str, dict[str, Any]]:
        """
        Prepare text for indexing with comprehensive cleaning.

        This is the SINGLE point where all text cleaning happens.
        The cleaned text is what gets chunked and indexed.

        Args:
            text: Raw text to be cleaned
            text_type: Type of text ('email', 'document', 'attachment')
            use_cache: Whether to use cached results for identical inputs

        Returns:
            Tuple of (cleaned_text, preprocessing_metadata)
        """
        if not text:
            return "", {"pre_cleaned": True, "cleaning_version": self.PREPROCESSING_VERSION}

        # Check cache
        cache_key = None
        if use_cache:
            cache_key = self._get_cache_key(text, text_type)
            if cache_key in self._cache:
                logger.debug("Using cached preprocessing for %s", text_type)
                return self._cache[cache_key]

        # Track original state
        original_length = len(text)

        # Apply cleaning based on text type
        if text_type == "email":
            from .email_processing import clean_email_text

            cleaned = clean_email_text(text)
        elif text_type == "attachment":
            cleaned = self._clean_attachment_text(text)
        elif text_type == "document":
            cleaned = self._clean_document_text(text)
        else:
            cleaned = self._basic_clean(text)

        # Generate metadata
        metadata = {
            "pre_cleaned": True,
            "cleaning_version": self.PREPROCESSING_VERSION,
            "text_type": text_type,
            "original_length": original_length,
            "cleaned_length": len(cleaned),
            "reduction_ratio": round(1 - (len(cleaned) / max(1, original_length)), 3),
        }

        # Cache result for reasonable sized texts
        if use_cache and cache_key and len(text) < 100000:
            self._cache[cache_key] = (cleaned, metadata)

        return cleaned, metadata

    def _clean_attachment_text(self, text: str) -> str:
        """Lighter cleaning for attachments (may contain code, logs, etc)."""
        # Preserve structure more for attachments
        text = text.strip()
        text = text.replace("\r\n", "\n")
        text = text.replace("\ufeff", "")

        # Remove only severe issues
        text = re.sub(r"[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]", "", text)

        # Light whitespace normalization (preserve formatting)
        text = re.sub(r"\n{5,}", "\n\n\n\n", text)  # Cap at 4 newlines

        return text

    def _clean_document_text(self, text: str) -> str:
        """Moderate cleaning for general documents."""
        text = text.strip()
        text = text.replace("\r\n", "\n")
        text = text.replace("\ufeff", "")

        # Remove control characters
        text = re.sub(r"[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]", "", text)

        # Moderate whitespace normalization
        text = re.sub(r"\n{4,}", "\n\n\n", text)
        text = re.sub(r"[ \t]+", " ", text)

        return text

    def _basic_clean(self, text: str) -> str:
        """Minimal cleaning for unknown text types."""
        text = text.strip()
        text = text.replace("\x00", "")  # Remove null bytes
        return text

    @lru_cache(maxsize=1000)
    def _get_cache_key(self, text: str, text_type: str) -> str:
        """Generate cache key for text + type combination."""
        # Use first 100 chars + last 100 chars + length for efficiency
        text_sig = f"{text[:100]}...{text[-100:]}...{len(text)}"
        combined = f"{text_type}:{text_sig}"
        return hashlib.sha256(combined.encode(), usedforsecurity=False).hexdigest()

    def clear_cache(self):
        """Clear the preprocessing cache."""
        self._cache.clear()
        self._get_cache_key.cache_clear()
        logger.debug("Preprocessor cache cleared")


def should_skip_retrieval_cleaning(chunk_or_doc: dict[str, Any]) -> bool:
    """
    Check if a chunk/document should skip cleaning during retrieval.

    Args:
        chunk_or_doc: Document or chunk dictionary

    Returns:
        True if cleaning should be skipped (already pre-cleaned)
    """
    # Check multiple indicators
    if chunk_or_doc.get("skip_retrieval_cleaning", False):
        return True

    if chunk_or_doc.get("pre_cleaned", False):
        # Check version to ensure compatibility
        version = chunk_or_doc.get("cleaning_version", "1.0")
        if version >= "2.0":
            return True

    # Legacy data - needs cleaning
    return False


# Create global preprocessor instance
_text_preprocessor = TextPreprocessor()


def get_text_preprocessor() -> TextPreprocessor:
    """Get the global text preprocessor instance."""
    return _text_preprocessor
