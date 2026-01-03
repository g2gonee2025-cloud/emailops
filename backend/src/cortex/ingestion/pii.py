"""
PII Detection and Redaction.

Implements §6.4 of the Canonical Blueprint:
- Wraps Presidio (or spaCy-based) PII detection
- Replaces detected entities with <<ENTITY>> placeholders
- Supports multiple entity types for comprehensive PII coverage
- Provides statistics on detected entities for auditing
"""

# pyright: reportMissingImports=false, reportOptionalCall=false
from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass, field
from typing import Any

from cortex.config.loader import get_config

logger = logging.getLogger(__name__)

_HAS_PRESIDIO = False

try:
    from presidio_analyzer import AnalyzerEngine as _AnalyzerEngine
    from presidio_analyzer import RecognizerResult as _RecognizerResult
    from presidio_anonymizer import AnonymizerEngine as _AnonymizerEngine
    from presidio_anonymizer.entities import OperatorConfig as _OperatorConfig

    _HAS_PRESIDIO = True
except ImportError:
    logger.warning(
        "Microsoft Presidio not found. PII redaction will be limited to regex-based fallback."
    )
    _AnalyzerEngine = None
    _AnonymizerEngine = None
    _OperatorConfig = None
    _RecognizerResult = None


# Entity types to detect per §6.4
SUPPORTED_ENTITIES = [
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "US_SSN",
    "CREDIT_CARD",
    "PERSON",  # Names via NER
    "LOCATION",  # Addresses/locations via NER
    "DATE_TIME",  # Dates that could be PII
    "IP_ADDRESS",  # IP addresses
    "URL",  # URLs (may contain PII)
    "US_DRIVER_LICENSE",
    "US_PASSPORT",
    "IBAN_CODE",
    "US_BANK_NUMBER",
]

# Placeholder mapping for each entity type
ENTITY_PLACEHOLDERS: dict[str, str] = {
    "EMAIL_ADDRESS": "<<EMAIL>>",
    "PHONE_NUMBER": "<<PHONE>>",
    "US_SSN": "<<SSN>>",
    "CREDIT_CARD": "<<CREDIT_CARD>>",
    "PERSON": "<<PERSON>>",
    "LOCATION": "<<LOCATION>>",
    "DATE_TIME": "<<DATE>>",
    "IP_ADDRESS": "<<IP>>",
    "URL": "<<URL>>",
    "US_DRIVER_LICENSE": "<<LICENSE>>",
    "US_PASSPORT": "<<PASSPORT>>",
    "IBAN_CODE": "<<IBAN>>",
    "US_BANK_NUMBER": "<<BANK_ACCOUNT>>",
    "DEFAULT": "<<PII>>",
}


@dataclass
class PIIDetectionResult:
    """Result of PII detection with statistics."""

    original_text: str
    redacted_text: str
    entities_found: dict[str, int] = field(default_factory=dict)
    total_entities: int = 0
    was_modified: bool = False

    @property
    def has_pii(self) -> bool:
        """Check if any PII was detected."""
        return self.total_entities > 0


@dataclass
class PIIEntity:
    """Represents a detected PII entity."""

    entity_type: str
    text: str
    start: int
    end: int
    score: float


class RegexPIIFallback:
    """
    Regex-based PII detection fallback when Presidio is unavailable.

    Provides basic detection for common PII patterns.
    """

    PATTERNS = {
        "EMAIL_ADDRESS": re.compile(
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", re.IGNORECASE
        ),
        "PHONE_NUMBER": re.compile(
            r"(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"
        ),
        "US_SSN": re.compile(r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b"),
        "CREDIT_CARD": re.compile(
            # mastercard, visa, amex, discover
            r"\b(?:4\d{12}(?:\d{3})?|5[1-5]\d{14}|3[47]\d{13}|6(?:011|5\d{2})\d{12})\b"
        ),
        "IP_ADDRESS": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    }

    def detect(self, text: str) -> list[PIIEntity]:
        """Detect PII using regex patterns."""
        entities: list[PIIEntity] = []
        for entity_type, pattern in self.PATTERNS.items():
            for match in pattern.finditer(text):
                entities.append(
                    PIIEntity(
                        entity_type=entity_type,
                        text=match.group(),
                        start=match.start(),
                        end=match.end(),
                        score=0.85,  # Fixed confidence for regex matches
                    )
                )
        return entities


class PIIInitError(Exception):
    """Raised when PII engine fails to initialize in strict mode."""

    pass


class PIIEngine:
    """
    PII Detection and Redaction Engine.

    Implements §6.4: wraps Presidio with fallback to regex-based detection.
    Replaces detected PII with <<ENTITY>> placeholders as required by blueprint.
    """

    def __init__(
        self,
        entities: list[str] | None = None,
        language: str = "en",
        score_threshold: float = 0.5,
        strict: bool = True,
    ):
        """
        Initialize PII engine.

        Args:
            entities: List of entity types to detect (defaults to SUPPORTED_ENTITIES)
            language: Language for detection (default: English)
            score_threshold: Minimum confidence score to consider a detection valid
            strict: If True, raise PIIInitError if Presidio fails to load.
        """
        self.entities = entities if entities is not None else SUPPORTED_ENTITIES
        self.language = language
        self.score_threshold = score_threshold
        self.analyzer: Any | None = None
        self.anonymizer: Any | None = None
        self._fallback = RegexPIIFallback()
        self._operators: dict[str, Any] = {}

        if _HAS_PRESIDIO and callable(_AnalyzerEngine) and callable(_AnonymizerEngine):
            try:
                self.analyzer = _AnalyzerEngine()
                self.anonymizer = _AnonymizerEngine()
                self._build_operators()
                logger.info(
                    f"PIIEngine initialized with Presidio. Entities: {len(self.entities)}"
                )
            except Exception as e:
                if strict:
                    raise PIIInitError(
                        f"Failed to initialize Presidio in strict mode: {e}"
                    ) from e
                logger.error(
                    "Failed to initialize Presidio: %s. Using regex fallback.", e
                )
                self.analyzer = None
                self.anonymizer = None
        else:
            if strict:
                raise PIIInitError("Presidio not installed but strict mode is enabled.")
            logger.info("PIIEngine using regex fallback (Presidio not available)")

    def _build_operators(self) -> None:
        """Build anonymization operators for each entity type."""
        if not _HAS_PRESIDIO or not callable(_OperatorConfig):
            return

        for entity_type in self.entities:
            placeholder = ENTITY_PLACEHOLDERS.get(
                entity_type, ENTITY_PLACEHOLDERS["DEFAULT"]
            )
            self._operators[entity_type] = _OperatorConfig(
                "replace", {"new_value": placeholder}
            )
        # Default operator for any unhandled entity types
        self._operators["DEFAULT"] = _OperatorConfig(
            "replace", {"new_value": ENTITY_PLACEHOLDERS["DEFAULT"]}
        )

    def detect(self, text: str) -> list[PIIEntity]:
        """
        Detect PII entities in text without redacting.

        Args:
            text: Text to analyze

        Returns:
            List of detected PII entities
        """
        if not text:
            return []

        if self.analyzer:
            try:
                results = self.analyzer.analyze(
                    text=text,
                    entities=self.entities,
                    language=self.language,
                    score_threshold=self.score_threshold,
                )
                return [
                    PIIEntity(
                        entity_type=r.entity_type,
                        text=text[r.start : r.end],
                        start=r.start,
                        end=r.end,
                        score=r.score,
                    )
                    for r in results
                ] + self._fallback.detect(text)
            except Exception as e:
                logger.warning(f"Presidio detection failed: {e}. Using fallback.")

        # Use regex fallback
        return self._fallback.detect(text)

    def redact(self, text: str) -> str:
        """
        Redact PII from text (simple interface).

        Replaces detected entities with placeholders like <<EMAIL>>, <<PHONE>>.

        Args:
            text: Text to redact

        Returns:
            Redacted text with PII replaced by placeholders
        """
        return self.redact_with_stats(text).redacted_text

    def _merge_fallback_entities(self, presidio_results: list, text: str) -> list:
        """Merge regex fallback entities into Presidio results, avoiding overlaps."""
        if not _RecognizerResult:
            return presidio_results

        regex_entities = self._fallback.detect(text)
        merged_results = list(presidio_results)

        for ent in regex_entities:
            # Check for overlap with existing results
            overlap = any((ent.start < r.end) and (ent.end > r.start) for r in merged_results)

            if not overlap:
                merged_results.append(
                    _RecognizerResult(
                        entity_type=ent.entity_type,
                        start=ent.start,
                        end=ent.end,
                        score=ent.score,
                    )
                )
        return merged_results

    def _redact_with_presidio(self, text: str) -> PIIDetectionResult:
        """Handle redaction using Presidio, with a fallback to regex."""
        try:
            # Presidio-based redaction
            results = self.analyzer.analyze(
                text=text,
                entities=self.entities,
                language=self.language,
                score_threshold=self.score_threshold,
            )

            # Merge with regex fallback entities
            all_results = self._merge_fallback_entities(results, text)

            if not all_results:
                return PIIDetectionResult(original_text=text, redacted_text=text)

            # Count entities
            entities_count = {}
            for r in all_results:
                entities_count[r.entity_type] = (
                    entities_count.get(r.entity_type, 0) + 1
                )

            # Anonymize
            anonymized = self.anonymizer.anonymize(
                text=text, analyzer_results=all_results, operators=self._operators
            )

            return PIIDetectionResult(
                original_text=text,
                redacted_text=anonymized.text,
                entities_found=entities_count,
                total_entities=len(all_results),
                was_modified=True,
            )
        except Exception as e:
            logger.error(f"PII redaction error with Presidio: {e}. Falling back to regex.")
            return self._redact_with_regex_fallback(text)

    def _redact_with_regex_fallback(self, text: str) -> PIIDetectionResult:
        """Handle redaction using only regex patterns."""
        entities = self._fallback.detect(text)
        if not entities:
            return PIIDetectionResult(original_text=text, redacted_text=text)

        # Sort by start position (descending) to replace from end to start
        # This preserves correct positions during replacement
        entities.sort(key=lambda e: e.start, reverse=True)

        redacted = text
        entities_count: dict[str, int] = {}
        for entity in entities:
            placeholder = ENTITY_PLACEHOLDERS.get(
                entity.entity_type, ENTITY_PLACEHOLDERS["DEFAULT"]
            )
            redacted = redacted[: entity.start] + placeholder + redacted[entity.end :]
            entities_count[entity.entity_type] = (
                entities_count.get(entity.entity_type, 0) + 1
            )

        return PIIDetectionResult(
            original_text=text,
            redacted_text=redacted,
            entities_found=entities_count,
            total_entities=len(entities),
            was_modified=True,
        )

    def redact_with_stats(self, text: str) -> PIIDetectionResult:
        """
        Redact PII from text and return detailed statistics.

        Args:
            text: Text to redact

        Returns:
            PIIDetectionResult with redacted text and entity counts
        """
        if not text:
            return PIIDetectionResult(
                original_text=text,
                redacted_text=text,
                was_modified=False,
            )

        if self.analyzer and self.anonymizer:
            return self._redact_with_presidio(text)

        # Regex fallback redaction
        return self._redact_with_regex_fallback(text)

    def is_available(self) -> bool:
        """Check if full PII detection (Presidio) is available."""
        return self.analyzer is not None

    def get_supported_entities(self) -> list[str]:
        """Get list of supported entity types."""
        return list(self.entities)


# Module-level singleton
_engine: PIIEngine | None = None
_pii_engine_lock = threading.Lock()


def get_pii_engine() -> PIIEngine:
    """Get the singleton PII engine instance."""
    global _engine
    if _engine is None:
        with _pii_engine_lock:
            if _engine is None:
                config = get_config()
                strict_cfg = getattr(config, "pii", None)
                strict_mode = strict_cfg.strict if strict_cfg is not None else True
                _engine = PIIEngine(strict=strict_mode)
    return _engine


def redact_pii(text: str) -> str:
    """
    Convenience function to redact PII from text.

    Args:
        text: Text to redact

    Returns:
        Text with PII replaced by placeholders
    """
    config = get_config()
    if not getattr(config.pii, "enabled", True):
        return text
    return get_pii_engine().redact(text)


def detect_pii(text: str) -> list[PIIEntity]:
    """
    Convenience function to detect PII in text.

    Args:
        text: Text to analyze

    Returns:
        List of detected PII entities
    """
    return get_pii_engine().detect(text)
