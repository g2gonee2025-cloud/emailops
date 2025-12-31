"""
CortexError hierarchy for EmailOps.

Provides specific, actionable exception types with context preservation
and programmatic error handling support.
"""

from __future__ import annotations

from typing import Any

SENSITIVE_CONTEXT_KEYS = {"raw_output", "query", "file_path"}
REDACTED_VALUE = "[REDACTED]"


def _redact_context(context: dict[str, Any]) -> dict[str, Any]:
    redacted: dict[str, Any] = {}
    for key, value in context.items():
        if isinstance(key, str) and key.lower() in SENSITIVE_CONTEXT_KEYS:
            redacted[key] = REDACTED_VALUE
        else:
            redacted[key] = value
    return redacted


def _pop_duplicate_kwargs(kwargs: dict[str, Any], keys: tuple[str, ...]) -> None:
    for key in keys:
        kwargs.pop(key, None)


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
        self.context = dict(context) if context is not None else {}
        if kwargs:
            self.context.update(kwargs)

    def to_dict(self) -> dict[str, Any]:
        """Serialize exception for logging/reporting."""
        safe_context = _redact_context(dict(self.context)) if self.context else {}
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "error_code": self.error_code,
            "context": safe_context,
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
        _pop_duplicate_kwargs(kwargs, ("retryable",))
        super().__init__(message, retryable=retryable, **kwargs)
        self.retryable = retryable


class ProcessingError(CortexError):
    """Document/text processing operation failures."""

    def __init__(
        self,
        message: str,
        retryable: bool = False,
        **kwargs: Any,
    ) -> None:
        _pop_duplicate_kwargs(kwargs, ("retryable",))
        super().__init__(message, retryable=retryable, **kwargs)
        self.retryable = retryable


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
        _pop_duplicate_kwargs(kwargs, ("field", "rule"))
        super().__init__(message, field=field, rule=rule, **kwargs)
        self.field = field
        self.rule = rule


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
        _pop_duplicate_kwargs(kwargs, ("provider", "retryable"))
        super().__init__(message, provider=provider, retryable=retryable, **kwargs)
        self.provider = provider
        self.retryable = retryable


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
        _pop_duplicate_kwargs(kwargs, ("file_path", "operation"))
        super().__init__(message, file_path=file_path, operation=operation, **kwargs)
        self.file_path = file_path
        self.operation = operation


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
        _pop_duplicate_kwargs(kwargs, ("transaction_id",))
        super().__init__(message, transaction_id=transaction_id, **kwargs)
        self.transaction_id = transaction_id


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
        _pop_duplicate_kwargs(kwargs, ("threat_type",))
        super().__init__(message, threat_type=threat_type, **kwargs)
        self.threat_type = threat_type


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
        _pop_duplicate_kwargs(kwargs, ("schema_name", "raw_output", "repair_attempts"))
        super().__init__(
            message,
            schema_name=schema_name,
            raw_output=raw_output,
            repair_attempts=repair_attempts,
            **kwargs,
        )
        self.schema_name = schema_name
        self.raw_output = raw_output
        self.repair_attempts = repair_attempts


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
        _pop_duplicate_kwargs(kwargs, ("query",))
        super().__init__(message, query=query, **kwargs)
        self.query = query


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
        _pop_duplicate_kwargs(kwargs, ("provider", "retry_after", "retryable"))
        super().__init__(
            message,
            provider=provider,
            retryable=True,
            retry_after=retry_after,
            **kwargs,
        )
        self.retry_after = retry_after


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
        _pop_duplicate_kwargs(kwargs, ("provider", "reset_at", "retryable"))
        super().__init__(
            message,
            provider=provider,
            retryable=False,
            reset_at=reset_at,
            **kwargs,
        )
        self.reset_at = reset_at


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
        _pop_duplicate_kwargs(kwargs, ("threat_type", "action", "policy_name"))
        super().__init__(
            message,
            threat_type="policy_violation",
            action=action,
            policy_name=policy_name,
            **kwargs,
        )
        self.action = action
        self.policy_name = policy_name
