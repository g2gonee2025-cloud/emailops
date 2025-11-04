"""Indexing operations - Direct Python API for building search indices.

Eliminates subprocess overhead, provides progress callbacks for long operations."""
from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from emailops.common.types import Result

log = logging.getLogger(__name__)


class IndexingError(Exception):
    """Raised when indexing operation fails."""

    pass


@dataclass(frozen=True)
class IndexingProgress:
    """Progress information for indexing operations."""

    stage: str  # "scanning", "chunking", "embedding", "saving", "complete"
    progress_pct: int  # 0-100
    message: str
    current: int = 0
    total: int = 0
    conversations_processed: int = 0
    documents_indexed: int = 0


def index_documents(
    root: Path,
    provider: str = "vertex",
    *,
    force_reindex: bool = False,
    limit: int | None = None,
    progress_callback: Callable[[IndexingProgress], None] | None = None,
) -> Result[dict[str, Any], str]:
    """
    Build or update search index with progress reporting.

    Direct Python API - no subprocess overhead. Enables progress callbacks
    for responsive GUI updates during long indexing operations.

    Args:
        root: Export root containing conversation directories
        provider: Embedding provider (default: "vertex")
        force_reindex: If True, rebuild entire index from scratch
        limit: Optional limit on chunks per conversation (for testing)
        progress_callback: Optional callback(IndexingProgress) for UI updates

    Returns:
        Result[dict, str] containing index statistics or error message

    Example:
        >>> def on_progress(p: IndexingProgress):
        ...     print(f"{p.stage}: {p.progress_pct}% - {p.message}")
        >>> result = index_documents(
        ...     Path("export"),
        ...     provider="vertex",
        ...     progress_callback=on_progress
        ... )
        >>> if result.ok:
        ...     stats = result.value
        ...     print(f"Indexed {stats['num_documents']} documents")
    """
    try:
        # Validate inputs
        if not root or not root.exists():
            return Result.failure(f"Export root not found: {root}")

        if provider != "vertex":
            return Result.failure(f"Only 'vertex' provider supported, got: {provider}")

        if force_reindex:
            log.debug("Force reindex requested for root %s", root)

        if limit is not None and limit <= 0:
            return Result.failure("limit must be a positive integer when provided")

        # Progress: Scanning conversations
        if progress_callback:
            progress_callback(
                IndexingProgress(
                    stage="scanning",
                    progress_pct=5,
                    message="Scanning conversation directories...",
                )
            )

        # Import indexing logic (deferred to avoid circular imports)

        # Note: This is a placeholder implementation
        # Full implementation requires refactoring indexing_main.py to:
        # 1. Accept parameters programmatically (not argparse)
        # 2. Emit progress events via callback
        # 3. Return structured results instead of sys.exit()

        # For now, document the integration pattern
        request_details = {
            "force_reindex": force_reindex,
            "limit": limit,
            "provider": provider,
        }
        return Result.failure(
            "Direct indexing API not yet implemented. "
            "Requires refactoring indexing_main.py to accept programmatic parameters. "
            "Use CLI for now: python -m emailops.indexing_main --root <path>. "
            f"Requested options: {request_details}"
        )

    except Exception as e:
        error_msg = f"Indexing failed: {e!s}"
        if progress_callback:
            progress_callback(
                IndexingProgress(stage="error", progress_pct=0, message=error_msg)
            )
        return Result.failure(error_msg)


__all__ = ["IndexingError", "IndexingProgress", "index_documents"]
