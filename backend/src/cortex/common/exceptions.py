"""
CortexError hierarchy for EmailOps.

Provides specific, actionable exception types with context preservation
and programmatic error handling support.
"""

from __future__ import annotations

from typing import Any


class CortexError(Exception):
    """
    Base class for all application errors.

    Attributes:
        message: Human-readable error description
        error_code: Machine-readable error code (e.g., "INDEX_CORRUPT")
        context: Additional context dict for debugging
    """

    def __init__(
        self,
        message: str,
        error_code: str | None = None,
        context: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.context = context or {}
        if kwargs:
            self.context.update(kwargs)

    def to_dict(self) -> dict[str, Any]:
        """Serialize exception for logging/reporting."""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "error_code": self.error_code,
            "context": self.context,
        }


class ConfigurationError(CortexError):
    """Configuration issues: missing/invalid settings."""


class SearchIndexError(CortexError):
    """
    Search index issues: not found, corrupted, incompatible.
    """


class EmbeddingError(CortexError):
    """
    Embedding operation failures: provider issues, dimension mismatches.
    """

    def __init__(self, message: str, retryable: bool = False, **kwargs: Any) -> None:
        self.retryable = kwargs.pop("retryable", retryable)
        super().__init__(message, **kwargs)


class ProcessingError(CortexError):
    """Document/text processing operation failures."""

    def __init__(
        self,
        message: str,
        retryable: bool = False,
        **kwargs: Any,
    ) -> None:
        self.retryable = kwargs.pop("retryable", retryable)
        super().__init__(message, **kwargs)


class ValidationError(CortexError):
    """
    Input validation failures.
    """

    def __init__(
        self,
        message: str,
        field: str | None = None,
        rule: str | None = None,
        **kwargs: Any,
    ) -> None:
        self.field = kwargs.pop("field", field)
        self.rule = kwargs.pop("rule", rule)
        super().__init__(message, **kwargs)


class ProviderError(CortexError):
    """
    External provider (LLM, embedding) operation failures.
    """

    def __init__(
        self,
        message: str,
        provider: str | None = None,
        retryable: bool = False,
        **kwargs: Any,
    ) -> None:
        self.provider = kwargs.pop("provider", provider)
        self.retryable = kwargs.pop("retryable", retryable)
        super().__init__(message, **kwargs)


class FileOperationError(CortexError):
    """
    File I/O operation failures.
    """

    def __init__(
        self,
        message: str,
        file_path: str | None = None,
        operation: str | None = None,
        **kwargs: Any,
    ) -> None:
        self.file_path = kwargs.pop("file_path", file_path)
        self.operation = kwargs.pop("operation", operation)
        super().__init__(message, **kwargs)


class TransactionError(CortexError):
    """
    Transaction operation failures (commit, rollback, recovery).
    """

    def __init__(
        self,
        message: str,
        transaction_id: str | None = None,
        **kwargs: Any,
    ) -> None:
        self.transaction_id = kwargs.pop("transaction_id", transaction_id)
        super().__init__(message, **kwargs)


class SecurityError(CortexError):
    """
    Security violations (injection attempts, unauthorized access).
    """

    def __init__(
        self,
        message: str,
        threat_type: str | None = None,
        **kwargs: Any,
    ) -> None:
        self.threat_type = kwargs.pop("threat_type", threat_type)
        super().__init__(message, **kwargs)


class LLMOutputSchemaError(CortexError):
    """
    LLM output did not match expected schema after repair attempts.

    Raised when complete_json cannot produce valid output matching the schema.
    """



    def __init__(
        self,
        message: str,
        schema_name: str | None = None,
        raw_output: str | None = None,
        repair_attempts: int = 0,
        **kwargs: Any,
    ) -> None:
        self.schema_name = kwargs.pop("schema_name", schema_name)
        self.raw_output = kwargs.pop("raw_output", raw_output)
        self.repair_attempts = kwargs.pop("repair_attempts", repair_attempts)
        super().__init__(message, **kwargs)


class RetrievalError(CortexError):
    """
    Search/retrieval operation failures.
    """

    def __init__(
        self,
        message: str,
        query: str | None = None,
        **kwargs: Any,
    ) -> None:
        self.query = kwargs.pop("query", query)
        super().__init__(message, **kwargs)


class RateLimitError(ProviderError):
    """
    Rate limit exceeded for external provider.

    Always retryable - includes retry_after hint if available.
    """

    def __init__(
        self,
        message: str,
        provider: str | None = None,
        retry_after: float | None = None,
        **kwargs: Any,
    ) -> None:
        self.retry_after = kwargs.pop("retry_after", retry_after)
        provider = kwargs.pop("provider", provider)
        kwargs.pop("retryable", None)
        super().__init__(message, provider=provider, retryable=True, **kwargs)


class CircuitBreakerOpenError(ProviderError):
    """
    Circuit breaker is open, requests are being rejected.
    """

    def __init__(
        self,
        message: str,
        provider: str | None = None,
        reset_at: float | None = None,
        **kwargs: Any,
    ) -> None:
        self.reset_at = kwargs.pop("reset_at", reset_at)
        provider = kwargs.pop("provider", provider)
        kwargs.pop("retryable", None)
        super().__init__(message, provider=provider, retryable=False, **kwargs)


class PolicyViolationError(SecurityError):
    """
    Action denied by policy enforcer.
    """

    def __init__(
        self,
        message: str,
        action: str | None = None,
        policy_name: str | None = None,
        **kwargs: Any,
    ) -> None:
        self.action = kwargs.pop("action", action)
        self.policy_name = kwargs.pop("policy_name", policy_name)
        kwargs.pop("threat_type", None)
        super().__init__(message, threat_type="policy_violation", **kwargs)
