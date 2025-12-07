"""
Attachment extraction.

Implements §6.5 of the Canonical Blueprint.
"""
from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

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
    mime_type: Optional[str]


class ExtractedAttachment(BaseModel):
    """
    Extracted attachment content.

    Blueprint §6.5:
    * text: str
    * tables: List[Dict[str, Any]]
    * metadata: Dict[str, Any]
    """

    text: str
    tables: List[Dict[str, Any]]
    metadata: Dict[str, Any]


def process_attachment(ref: AttachmentRef) -> Optional[ExtractedAttachment]:
    """
    Process a single attachment.

    Uses cortex.text_extraction to extract text and metadata.
    """
    try:
        text = extract_text(Path(ref.path))
        return ExtractedAttachment(
            text=text, tables=[], metadata={"source_path": ref.path}
        )
    except Exception as e:
        logger.error(f"Failed to process attachment {ref.path}: {e}")
        return None
