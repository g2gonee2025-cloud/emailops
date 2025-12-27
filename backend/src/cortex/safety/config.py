"""
Safety Configuration.
"""
from __future__ import annotations

import re
from typing import List, Set

from pydantic import BaseModel, Field


class PolicyConfig(BaseModel):
    """
    Defines the security and operational policies for actions.
    """
    # Risk levels for actions
    low_risk_actions: Set[str] = Field(default_factory=lambda: {
        "search", "read_thread", "read_message", "get_thread_context",
        "summarize_thread", "answer_question"
    })
    medium_risk_actions: Set[str] = Field(default_factory=lambda: {
        "draft_email", "create_draft", "modify_draft", "upload_attachment"
    })
    high_risk_actions: Set[str] = Field(default_factory=lambda: {
        "send_email", "delete_message", "delete_thread", "delete_attachment",
        "admin_action", "export_data", "bulk_operation"
    })

    # Recipient policies
    external_domain_pattern: str = r"@(?!internal\.company\.com$)"
    max_recipients_auto_approve: int = 10

    # Sensitive content patterns (using more generic regex)
    sensitive_patterns: List[str] = Field(default_factory=lambda: [
        r"\b(confidential|secret|private|internal only)\b",
        r"\b(password|credential|api.?key|token)\b",
        r"\b\d{3}[- ]?\d{2}[- ]?\d{4}\b",  # SSN (with optional separators)
        r"\b(?:(?:\d{4}[- ]?){3}\d{4}|\d{13,19})\b",  # Common credit card formats
    ])

    # Attachment policies
    dangerous_extensions: Set[str] = Field(default_factory=lambda: {
        ".exe", ".bat", ".cmd", ".ps1", ".vbs", ".js"
    })
    max_attachment_size_mb: int = 25

    def get_sensitive_patterns(self) -> List[re.Pattern]:
        """Compile and return sensitive patterns as regex objects."""
        return [re.compile(p, re.IGNORECASE) for p in self.sensitive_patterns]

    def get_external_domain_pattern(self) -> re.Pattern:
        """Compile and return the external domain pattern."""
        return re.compile(self.external_domain_pattern, re.IGNORECASE)
