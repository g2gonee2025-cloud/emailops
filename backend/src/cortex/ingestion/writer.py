"""
Database Writer.

Implements §6.6 of the Canonical Blueprint.
Ensures all chunks have:
* content_hash (for deduplication)
* pre_cleaned flag (text was cleaned before chunking)
* cleaning_version (version of cleaner used)
* source (e.g., 'email_body', 'attachment')
"""
from __future__ import annotations

import logging
from typing import Any, Dict

from cortex.db.models import Attachment, Chunk, Message, Thread
from cortex.ingestion.constants import CLEANING_VERSION
from cortex.ingestion.models import IngestJobRequest
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


def compute_content_hash(text: str) -> str:
    """Compute SHA-256 hash of content for deduplication."""
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def ensure_chunk_metadata(
    chunk_data: Dict[str, Any],
    source: str = "email_body",
    pre_cleaned: bool = True,
) -> Dict[str, Any]:
    """
    Ensure chunk has all required metadata fields per §7.1.

    Args:
        chunk_data: Raw chunk data dictionary
        source: Source type ('email_body', 'attachment', 'quoted_block')
        pre_cleaned: Whether text was pre-cleaned

    Returns:
        Updated chunk data with guaranteed metadata fields
    """
    metadata = dict(chunk_data.get("metadata", {}) or {})

    # Always compute content_hash if not present
    if "content_hash" not in metadata:
        text = chunk_data.get("text", "")
        metadata["content_hash"] = compute_content_hash(text)

    # Always set cleaning metadata
    metadata["pre_cleaned"] = pre_cleaned
    metadata["cleaning_version"] = CLEANING_VERSION

    # Set source if not present
    if "source" not in metadata:
        metadata["source"] = source

    chunk_data["metadata"] = metadata
    return chunk_data


class DBWriter:
    """
    Transactional writer for ingestion results.

    Handles writing:
    * Threads
    * Messages
    * Attachments
    * Chunks
    """

    def __init__(self, session: Session):
        self.session = session

    def write_job_results(self, job: IngestJobRequest, results: Dict[str, Any]) -> None:
        """
        Write ingestion results to DB.

        Ensures:
        * Consistent tenant_id
        * Referential integrity
        * FTS updates
        """
        try:
            # 1. Write threads
            for thread_data in results.get("threads", []):
                thread = Thread(
                    thread_id=thread_data["thread_id"],
                    tenant_id=job.tenant_id,
                    subject_norm=thread_data.get("subject_norm"),
                    original_subject=thread_data.get("original_subject"),
                    created_at=thread_data["created_at"],
                    updated_at=thread_data["updated_at"],
                    metadata_=thread_data.get("metadata", {}),
                )
                self.session.merge(thread)  # Upsert

            # 2. Write messages
            for msg_data in results.get("messages", []):
                message = Message(
                    message_id=msg_data["message_id"],
                    thread_id=msg_data["thread_id"],
                    tenant_id=job.tenant_id,
                    folder=msg_data.get("folder"),
                    sent_at=msg_data.get("sent_at"),
                    recv_at=msg_data.get("recv_at"),
                    from_addr=msg_data["from_addr"],
                    to_addrs=msg_data.get("to_addrs", []),
                    cc_addrs=msg_data.get("cc_addrs", []),
                    bcc_addrs=msg_data.get("bcc_addrs", []),
                    subject=msg_data.get("subject"),
                    body_plain=msg_data.get("body_plain"),
                    body_html=msg_data.get("body_html"),
                    has_quoted_mask=msg_data.get("has_quoted_mask", False),
                    raw_storage_uri=msg_data.get("raw_storage_uri"),
                    metadata_=msg_data.get("metadata", {}),
                )
                self.session.merge(message)

            # 3. Write attachments
            for att_data in results.get("attachments", []):
                attachment = Attachment(
                    attachment_id=att_data["attachment_id"],
                    message_id=att_data["message_id"],
                    tenant_id=job.tenant_id,
                    filename=att_data.get("filename"),
                    mime_type=att_data.get("mime_type"),
                    storage_uri_raw=att_data.get("storage_uri_raw"),
                    storage_uri_extracted=att_data.get("storage_uri_extracted"),
                    status=att_data.get("status", "pending"),
                    extracted_chars=att_data.get("extracted_chars", 0),
                    metadata_=att_data.get("metadata", {}),
                )
                self.session.merge(attachment)

            # 4. Write chunks - ensure all have canonical metadata
            for chunk_data in results.get("chunks", []):
                # Determine source from chunk type
                chunk_type = chunk_data.get("chunk_type", "unknown")
                if chunk_data.get("attachment_id"):
                    source = "attachment"
                elif chunk_type in {"quoted", "quoted_history"}:
                    source = "quoted_block"
                else:
                    source = "email_body"

                # Ensure canonical metadata fields
                chunk_data = ensure_chunk_metadata(
                    chunk_data,
                    source=source,
                    pre_cleaned=chunk_data.get("pre_cleaned", True),
                )

                chunk = Chunk(
                    chunk_id=chunk_data["chunk_id"],
                    thread_id=chunk_data["thread_id"],
                    message_id=chunk_data.get("message_id"),
                    attachment_id=chunk_data.get("attachment_id"),
                    tenant_id=job.tenant_id,
                    chunk_type=chunk_data["chunk_type"],
                    text=chunk_data["text"],
                    summary=chunk_data.get("summary"),
                    section_path=chunk_data["section_path"],
                    position=chunk_data["position"],
                    char_start=chunk_data["char_start"],
                    char_end=chunk_data["char_end"],
                    embedding=chunk_data.get("embedding"),
                    embedding_model=chunk_data.get("embedding_model"),
                    metadata_=chunk_data.get("metadata", {}),
                )
                self.session.merge(chunk)

            self.session.commit()
            logger.info(f"Successfully wrote job results for {job.job_id}")

        except Exception as e:
            self.session.rollback()
            logger.error(f"Failed to write job results: {e}")
            raise

    def write_chunks_with_dedup(
        self,
        chunks: list[Dict[str, Any]],
        tenant_id: str,
        source: str = "email_body",
    ) -> tuple[int, int]:
        """
        Write chunks with deduplication based on content_hash.

        Args:
            chunks: List of chunk data dictionaries
            tenant_id: Tenant ID for RLS
            source: Source type for metadata

        Returns:
            Tuple of (written_count, skipped_count)
        """
        written = 0
        skipped = 0

        for chunk_data in chunks:
            # Ensure metadata
            chunk_data = ensure_chunk_metadata(chunk_data, source=source)
            content_hash = chunk_data["metadata"]["content_hash"]

            # Check for existing chunk with same hash
            existing = (
                self.session.query(Chunk)
                .filter(Chunk.tenant_id == tenant_id)
                .filter(Chunk.metadata_["content_hash"].astext == content_hash)
                .first()
            )

            if existing:
                logger.debug(f"Skipping duplicate chunk: {content_hash[:16]}...")
                skipped += 1
                continue

            chunk = Chunk(
                chunk_id=chunk_data["chunk_id"],
                thread_id=chunk_data["thread_id"],
                message_id=chunk_data.get("message_id"),
                attachment_id=chunk_data.get("attachment_id"),
                tenant_id=tenant_id,
                chunk_type=chunk_data["chunk_type"],
                text=chunk_data["text"],
                summary=chunk_data.get("summary"),
                section_path=chunk_data["section_path"],
                position=chunk_data["position"],
                char_start=chunk_data["char_start"],
                char_end=chunk_data["char_end"],
                embedding=chunk_data.get("embedding"),
                embedding_model=chunk_data.get("embedding_model"),
                metadata_=chunk_data.get("metadata", {}),
            )
            self.session.add(chunk)
            written += 1

        self.session.commit()
        logger.info(f"Wrote {written} chunks, skipped {skipped} duplicates")
        return written, skipped
