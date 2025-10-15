"""
Centralized exception definitions for EmailOps.

This module provides consistent exception types across the codebase to improve
error handling and maintainability.
"""
from __future__ import annotations


class EmailOpsError(Exception):
    """Base exception for all EmailOps-related errors."""
    pass


class ConfigurationError(EmailOpsError):
    """Raised when there are configuration issues (missing settings, invalid values, etc.)."""
    pass


class IndexError(EmailOpsError):
    """Raised when there are issues with the search index (not found, corrupted, etc.)."""
    pass


class EmbeddingError(EmailOpsError):
    """Raised when embedding operations fail (provider issues, dimension mismatches, etc.)."""
    pass


class ProcessingError(EmailOpsError):
    """Raised when document/text processing operations fail."""
    pass


class ValidationError(EmailOpsError):
    """Raised when input validation fails."""
    pass


class ProviderError(EmailOpsError):
    """Raised when external provider (LLM, embedding) operations fail."""
    pass


class FileOperationError(EmailOpsError):
    """Raised when file I/O operations fail."""
    pass


# Backward compatibility aliases
LLMError = ProviderError  # For existing code using LLMError
ProcessorError = ProcessingError  # For existing code using ProcessorError
CommandExecutionError = ProcessingError  # For existing code
IndexNotFoundError = IndexError  # For existing code


__all__ = [
    "EmailOpsError",
    "ConfigurationError",
    "IndexError",
    "EmbeddingError",
    "ProcessingError",
    "ValidationError",
    "ProviderError",
    "FileOperationError",
    # Backward compatibility
    "LLMError",
    "ProcessorError",
    "CommandExecutionError",
    "IndexNotFoundError",
]
