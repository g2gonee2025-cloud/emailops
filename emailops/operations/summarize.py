"""Summarization operations - Direct Python API for conversation summarization.

Provides async summarization with progress tracking."""
from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from emailops.common.types import Result

log = logging.getLogger(__name__)


class SummarizationError(Exception):
    """Raised when summarization operation fails."""

    pass


@dataclass(frozen=True)
class SummarizationProgress:
    """Progress information for summarization operations."""

    stage: str  # "loading", "analyzing", "generating", "saving", "complete"
    progress_pct: int  # 0-100
    message: str
    current: int = 0
    total: int = 0


def summarize_conversation(
    conversation_dir: Path,
    provider: str = "vertex",
    *,
    output_dir: Path | None = None,
    progress_callback: Callable[[SummarizationProgress], None] | None = None,
) -> Result[dict[str, Any], str]:
    """
    Summarize a single conversation with progress reporting.

    Direct Python API - no subprocess overhead. Provides progress callbacks
    for responsive UI updates.

    Args:
        conversation_dir: Path to conversation directory
        provider: LLM provider for summarization (default: "vertex")
        output_dir: Optional output directory (defaults to conversation_dir)
        progress_callback: Optional callback(SummarizationProgress) for UI updates

    Returns:
        Result[dict, str] containing analysis data or error message

    Example:
        >>> def on_progress(p: SummarizationProgress):
        ...     print(f"{p.stage}: {p.progress_pct}%")
        >>> result = summarize_conversation(
        ...     Path("export/conv123"),
        ...     progress_callback=on_progress
        ... )
        >>> if result.ok:
        ...     analysis = result.value
        ...     print(analysis["summary"])
    """
    try:
        # Validate inputs
        if not conversation_dir or not conversation_dir.exists():
            return Result.failure(
                f"Conversation directory not found: {conversation_dir}"
            )

        conv_file = conversation_dir / "Conversation.txt"
        if not conv_file.exists():
            return Result.failure(
                f"Conversation.txt not found in {conversation_dir}"
            )

        # Progress: Loading conversation
        if progress_callback:
            progress_callback(
                SummarizationProgress(
                    stage="loading",
                    progress_pct=10,
                    message="Loading conversation...",
                )
            )

        # Note: This is a placeholder implementation
        # Full implementation requires refactoring feature_summarize.py to:
        # 1. Accept parameters programmatically (not just async wrapper)
        # 2. Emit progress events via callback
        # 3. Return structured results instead of writing files directly

        if output_dir and not output_dir.exists():
            log.debug("Creating output directory for summaries: %s", output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        request_details = {
            "provider": provider,
            "output_dir": str(output_dir) if output_dir else None,
        }
        return Result.failure(
            "Direct summarization API not yet implemented. "
            "Requires refactoring feature_summarize.py for programmatic access. "
            "Use CLI for now: python -m emailops.cli summarize --conversation <path>. "
            f"Requested options: {request_details}"
        )

    except Exception as e:
        error_msg = f"Summarization failed: {e!s}"
        if progress_callback:
            progress_callback(
                SummarizationProgress(stage="error", progress_pct=0, message=error_msg)
            )
        return Result.failure(error_msg)


__all__ = ["SummarizationError", "SummarizationProgress", "summarize_conversation"]
