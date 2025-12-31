"""
Attachment extraction.

Implements §6.5 of the Canonical Blueprint.
"""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from cortex.config.loader import get_config
from cortex.ingestion.text_preprocessor import get_text_preprocessor
from cortex.text_extraction import extract_text
from pydantic import BaseModel

logger = logging.getLogger(__name__)
_ATTACHMENT_EXECUTOR = ThreadPoolExecutor(
    max_workers=max(2, min(8, (os.cpu_count() or 4)))
)


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
    * tables: list[dict[str, Any]]
    * metadata: dict[str, Any]
    """

    text: str
    tables: list[dict[str, Any]]
    metadata: dict[str, Any]


async def extract_attachment_text(
    ref: AttachmentRef,
    path: Path,
    *,
    max_chars: int,
    skip_if_attachment_over_mb: float,
) -> str | None:
    """Extract text from an attachment with size/char limits."""
    attachment_id = ref.attachment_id
    try:
        if skip_if_attachment_over_mb and skip_if_attachment_over_mb > 0:
            try:
                loop = asyncio.get_running_loop()
                # Run synchronous file I/O in a separate thread to avoid blocking
                # the event loop.
                stat_result = await loop.run_in_executor(
                    _ATTACHMENT_EXECUTOR, path.stat
                )
                mb = stat_result.st_size / (1024 * 1024)
                if mb > skip_if_attachment_over_mb:
                    logger.info(
                        (
                            "Skipping large attachment %.2f MB > %.2f MB "
                            "[attachment_id: %s]"
                        ),
                        mb,
                        skip_if_attachment_over_mb,
                        attachment_id,
                    )
                    return None
            except FileNotFoundError:
                logger.warning(
                    "Attachment not found [attachment_id: %s]", attachment_id
                )
                return None
            except OSError as e:
                logger.error(
                    (
                        "OS error while checking attachment size for "
                        "[attachment_id: %s]: %s"
                    ),
                    attachment_id,
                    e,
                )
                return None

        loop = asyncio.get_running_loop()
        # Run the CPU-bound text extraction in a separate thread.
        text = await loop.run_in_executor(
            _ATTACHMENT_EXECUTOR, extract_text, path, max_chars
        )
        if text and text.strip():
            return text
    except FileNotFoundError:
        logger.error("Attachment file not found [attachment_id: %s]", attachment_id)
    except OSError as e:
        logger.error(
            "IO error processing attachment [attachment_id: %s]: %s", attachment_id, e
        )
    except Exception:
        logger.exception(
            "Unexpected error while processing attachment [attachment_id: %s]",
            attachment_id,
        )
        raise

    return None


async def process_attachment(
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
    loop = asyncio.get_running_loop()

    # Security: Prevent path traversal attacks.
    # Ensure the resolved path is within the secure upload directory.
    try:
        upload_dir = await loop.run_in_executor(
            _ATTACHMENT_EXECUTOR,
            lambda: Path(config.directories.local_upload_dir).resolve(strict=True),
        )
        attachment_path = await loop.run_in_executor(
            _ATTACHMENT_EXECUTOR, lambda: Path(ref.path).resolve(strict=True)
        )

        if attachment_path == upload_dir or upload_dir not in attachment_path.parents:
            logger.critical(
                "Path traversal attempt [attachment_id: %s]",
                ref.attachment_id,
            )
            return None
        if not attachment_path.is_file():
            logger.warning(
                "Attachment path is not a file [attachment_id: %s]",
                ref.attachment_id,
            )
            return None
    except (TypeError, ValueError, FileNotFoundError, AttributeError):
        logger.warning(
            "Invalid or missing attachment path: %s", ref.path, exc_info=True
        )
        return None

    limits = getattr(config, "limits", None)
    if limits is None:
        logger.error("Attachment limits configuration missing")
        return None
    text = await extract_attachment_text(
        ref,
        attachment_path,
        max_chars=(
            max_chars if max_chars is not None else limits.max_attachment_text_chars
        ),
        skip_if_attachment_over_mb=(
            skip_if_attachment_over_mb
            if skip_if_attachment_over_mb is not None
            else limits.skip_attachment_over_mb
        ),
    )
    if text is None:
        return None

    preprocessor = get_text_preprocessor()
    if not preprocessor or not hasattr(preprocessor, "prepare_for_indexing"):
        logger.error("Attachment preprocessor unavailable")
        return None

    cleaned_text, meta = preprocessor.prepare_for_indexing(
        text,
        text_type="attachment",
        tenant_id=tenant_id,
        metadata={"source_name": attachment_path.name},
    )

    return ExtractedAttachment(text=cleaned_text, tables=[], metadata=meta)
