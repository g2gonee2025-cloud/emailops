"""
P1-10 FIX: Enhanced exception hierarchy for EmailOps.

Provides specific, actionable exception types with context preservation
and programmatic error handling support.
"""

from __future__ import annotations

from typing import Any


class EmailOpsError(Exception):
    """
    Base exception for all EmailOps-related errors.

    P1-10 FIX: Enhanced with context preservation and error codes.

    Attributes:
        message: Human-readable error description
        error_code: Machine-readable error code (e.g., "INDEX_CORRUPT")
        context: Additional context dict for debugging
    """

    def __init__(self, message: str, error_code: str | None = None, context: dict[str, Any] | None = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.context = context or {}

    def to_dict(self) -> dict[str, Any]:
        """Serialize exception for logging/reporting."""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "error_code": self.error_code,
            "context": self.context,
        }


class ConfigurationError(EmailOpsError):
    """Configuration issues: missing/invalid settings."""
    pass


class EmailOpsIndexError(EmailOpsError):
    """
    Search index issues: not found, corrupted, incompatible.

    P1-10 FIX: Renamed to avoid shadowing Python's built-in IndexError.
    """
    pass


class EmbeddingError(EmailOpsError):
    """
    Embedding operation failures: provider issues, dimension mismatches.

    P1-10 FIX: Now distinguishes temporary vs permanent failures.
    """

    def __init__(self, message: str, retryable: bool = False, **kwargs):
        super().__init__(message, **kwargs)
        self.retryable = retryable


class ProcessingError(EmailOpsError):
    """Document/text processing operation failures."""
    pass


class ValidationError(EmailOpsError):
    """
    Input validation failures.

    P1-10 FIX: Includes field name and validation rule for programmatic handling.
    """

    def __init__(self, message: str, field: str | None = None, rule: str | None = None, **kwargs):
        super().__init__(message, **kwargs)
        self.field = field
        self.rule = rule


class ProviderError(EmailOpsError):
    """
    External provider (LLM, embedding) operation failures.

    P1-10 FIX: Distinguishes quota exhaustion, rate limits, auth failures.
    """

    def __init__(self, message: str, provider: str | None = None, retryable: bool = False, **kwargs):
        super().__init__(message, **kwargs)
        self.provider = provider
        self.retryable = retryable


class FileOperationError(EmailOpsError):
    """
    File I/O operation failures.

    P1-10 FIX: Includes file path and operation type for debugging.
    """

    def __init__(self, message: str, file_path: str | None = None, operation: str | None = None, **kwargs):
        super().__init__(message, **kwargs)
        self.file_path = file_path
        self.operation = operation


class TransactionError(EmailOpsError):
    """
    P0-4: Transaction operation failures (commit, rollback, recovery).

    New exception type for transactional index operations.
    """

    def __init__(self, message: str, transaction_id: str | None = None, **kwargs):
        super().__init__(message, **kwargs)
        self.transaction_id = transaction_id


class SecurityError(EmailOpsError):
    """
    P0-7: Security violations (injection attempts, unauthorized access).

    New exception type for security-related failures.
    """

    def __init__(self, message: str, threat_type: str | None = None, **kwargs):
        super().__init__(message, **kwargs)
        self.threat_type = threat_type


# Backward compatibility aliases
class LLMError(ProviderError):
    """Specific exception for LLM-related errors, inheriting from ProviderError."""
    pass

ProcessorError = ProcessingError
CommandExecutionError = ProcessingError
IndexNotFoundError = EmailOpsIndexError
EmailEmailIndexError = EmailOpsIndexError  # Fix typo from original
IndexError = EmailOpsIndexError  # Backward compat (shadows builtin but maintains API)  # noqa: A001

__all__ = [
    "CommandExecutionError",
    "ConfigurationError",
    "EmailEmailIndexError",
    "EmailOpsError",
    "EmbeddingError",
    "FileOperationError",
    "IndexError",
    "IndexNotFoundError",
    "LLMError",
    "ProcessingError",
    "ProcessorError",
    "ProviderError",
    "SecurityError",
    "TransactionError",
    "ValidationError",
]
