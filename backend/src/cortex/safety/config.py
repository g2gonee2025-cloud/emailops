"""
Safety Configuration.
"""

from __future__ import annotations

import re
from functools import lru_cache

from cortex.common.exceptions import ConfigurationError
from pydantic import BaseModel, ConfigDict, Field


class PolicyConfig(BaseModel):
    """
    Defines the security and operational policies for actions.
    """

    model_config = ConfigDict(validate_assignment=True)

    # Risk levels for actions
    low_risk_actions: set[str] = Field(
        default_factory=lambda: {
            "search",
            "read_thread",
            "read_message",
            "get_thread_context",
            "summarize_thread",
            "answer_question",
        }
    )
    medium_risk_actions: set[str] = Field(
        default_factory=lambda: {
            "draft_email",
            "create_draft",
            "modify_draft",
            "upload_attachment",
        }
    )
    high_risk_actions: set[str] = Field(
        default_factory=lambda: {
            "send_email",
            "delete_message",
            "delete_thread",
            "delete_attachment",
            "admin_action",
            "export_data",
            "bulk_operation",
        }
    )

    # Recipient policies
    external_domain_pattern: str = (
        r"@(?!(?:[A-Z0-9-]+\.)*internal\.company\.com(?=$|[^A-Z0-9.-]))"
        r"[A-Z0-9.-]+(?=$|[^A-Z0-9.-])"
    )
    max_recipients_auto_approve: int = 10

    # Sensitive content patterns (using more generic regex)
    sensitive_patterns: list[str] = Field(
        default_factory=lambda: [
            r"\b(confidential|secret|private|internal only)\b",
            r"\b(password|credential|api.?key|token)\b",
            r"\b\d{3}[- ]?\d{2}[- ]?\d{4}\b",  # SSN (with optional separators)
            r"\b(?:(?:\d{4}[- ]?){3}\d{4}|\d{13,19})\b",  # Common credit card formats
        ]
    )

    # Attachment policies
    dangerous_extensions: set[str] = Field(
        default_factory=lambda: {".exe", ".bat", ".cmd", ".ps1", ".vbs", ".js"}
    )
    max_attachment_size_mb: int = 25

    def get_sensitive_patterns(self) -> list[re.Pattern]:
        """Compile and return sensitive patterns as regex objects."""
        patterns = tuple(self.sensitive_patterns or [])
        if not patterns:
            return []
        return list(_compile_sensitive_patterns(patterns))

    def get_external_domain_pattern(self) -> re.Pattern:
        """Compile and return the external domain pattern."""
        return _compile_external_domain_pattern(self.external_domain_pattern)


@lru_cache(maxsize=128)
def _compile_sensitive_patterns(patterns: tuple[str, ...]) -> tuple[re.Pattern, ...]:
    compiled: list[re.Pattern] = []
    for pattern in patterns:
        try:
            compiled.append(re.compile(pattern, re.IGNORECASE))
        except re.error as exc:
            raise ConfigurationError(
                "Invalid sensitive pattern regex.",
                context={"pattern": pattern},
            ) from exc
    return tuple(compiled)


@lru_cache(maxsize=128)
def _compile_external_domain_pattern(pattern: str) -> re.Pattern:
    try:
        return re.compile(pattern, re.IGNORECASE)
    except re.error as exc:
        raise ConfigurationError(
            "Invalid external domain pattern regex.",
            context={"pattern": pattern},
        ) from exc
