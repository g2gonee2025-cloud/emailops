"""
Attachment extraction.

Implements §6.5 of the Canonical Blueprint.
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any

from cortex.config.loader import get_config
from cortex.ingestion.text_preprocessor import get_text_preprocessor
from cortex.text_extraction import extract_text
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class AttachmentRef(BaseModel):
    """
    Attachment reference.

    Blueprint §6.5:
    * attachment_id: UUID
    * message_id: str
    * path: str
    * mime_type: Optional[str]
    """

    attachment_id: uuid.UUID
    message_id: str
    path: str
    mime_type: str | None


class ExtractedAttachment(BaseModel):
    """
    Extracted attachment content.

    Blueprint §6.5:
    * text: str
    * tables: List[Dict[str, Any]]
    * metadata: Dict[str, Any]
    """

    text: str
    tables: list[dict[str, Any]]
    metadata: dict[str, Any]


def extract_attachment_text(
    path: Path,
    *,
    max_chars: int,
    skip_if_attachment_over_mb: float,
) -> str | None:
    """Extract text from an attachment with size/char limits."""

    try:
        if skip_if_attachment_over_mb and skip_if_attachment_over_mb > 0:
            try:
                mb = path.stat().st_size / (1024 * 1024)
                if mb > skip_if_attachment_over_mb:
                    logger.info(
                        "Skipping large attachment (%.2f MB > %.2f MB): %s",
                        mb,
                        skip_if_attachment_over_mb,
                        path,
                    )
                    return None
            except OSError:
                pass

        text = extract_text(path, max_chars=max_chars)
        if text and text.strip():
            return text
    except Exception as exc:  # broad but logs for observability
        logger.error("Failed to extract attachment %s: %s", path, exc)
    return None


def process_attachment(
    ref: AttachmentRef,
    *,
    tenant_id: str,
    max_chars: int | None = None,
    skip_if_attachment_over_mb: float | None = None,
) -> ExtractedAttachment | None:
    """
    Process a single attachment with limits and PII-safe cleaning.
    """
    config = get_config()
    limits = config.limits
    text = extract_attachment_text(
        Path(ref.path),
        max_chars=max_chars or limits.max_attachment_text_chars,
        skip_if_attachment_over_mb=(
            skip_if_attachment_over_mb
            if skip_if_attachment_over_mb is not None
            else limits.skip_attachment_over_mb
        ),
    )
    if text is None:
        return None

    cleaned_text, meta = get_text_preprocessor().prepare_for_indexing(
        text,
        text_type="attachment",
        tenant_id=tenant_id,
        metadata={"source_path": ref.path},
    )

    return ExtractedAttachment(text=cleaned_text, tables=[], metadata=meta)
