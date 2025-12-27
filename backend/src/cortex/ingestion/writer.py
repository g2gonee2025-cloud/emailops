"""
Database Writer.

Implements §6.6 of the Canonical Blueprint.
Updated for new Conversation-based schema.
"""

from __future__ import annotations

import hashlib
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

from cortex.db.models import Attachment, Chunk, Conversation
from cortex.ingestion.constants import CLEANING_VERSION
from cortex.ingestion.models import IngestJobRequest
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


def compute_content_hash(text: str) -> str:
    """Compute SHA-256 hash of content for deduplication."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def ensure_chunk_metadata(
    chunk_data: dict[str, Any],
    source: str = "email_body",
    pre_cleaned: bool = True,
) -> dict[str, Any]:
    """
    Ensure chunk has all required metadata fields per §7.1.
    """
    extra_data = dict(chunk_data.get("extra_data", {}) or {})

    # Always compute content_hash if not present
    if "content_hash" not in extra_data:
        text = chunk_data.get("text", "")
        extra_data["content_hash"] = compute_content_hash(text)

    # Always set cleaning metadata
    extra_data["pre_cleaned"] = pre_cleaned
    extra_data["cleaning_version"] = CLEANING_VERSION
    extra_data["source"] = source

    chunk_data["extra_data"] = extra_data
    return chunk_data


@dataclass
class ChunkRecord:
    """DTO for chunk writing to avoid too many arguments."""

    tenant_id: str
    chunk_id: uuid.UUID
    conversation_id: uuid.UUID
    text: str
    chunk_type: str = "message_body"
    is_summary: bool = False
    embedding: list[float] | None = None
    position: int = 0
    char_start: int = 0
    char_end: int = 0
    section_path: str | None = None
    is_attachment: bool = False
    attachment_id: uuid.UUID | None = None
    extra_data: dict[str, Any] | None = None


class DBWriter:
    """
    Transactional writer for ingestion results.

    Handles writing:
    * Conversations
    * Attachments
    * Chunks
    """

    def __init__(self, session: Session):
        self.session = session

    def write_conversation(
        self,
        *,
        conversation_id: uuid.UUID,
        tenant_id: str,
        folder_name: str,
        subject: str | None = None,
        smart_subject: str | None = None,
        summary_text: str | None = None,
        participants: list[dict[str, Any]] | None = None,
        messages: list[dict[str, Any]] | None = None,
        earliest_date: datetime | None = None,
        latest_date: datetime | None = None,
        storage_uri: str | None = None,
        extra_data: dict[str, Any] | None = None,
    ) -> Conversation:
        """Write a conversation record."""
        conv = Conversation(
            conversation_id=conversation_id,
            tenant_id=tenant_id,
            folder_name=folder_name,
            subject=subject,
            smart_subject=smart_subject,
            summary_text=summary_text,
            participants=participants,
            messages=messages,
            earliest_date=earliest_date,
            latest_date=latest_date,
            storage_uri=storage_uri,
            extra_data=extra_data,
        )
        self.session.merge(conv)
        return conv

    def write_attachment(
        self,
        *,
        attachment_id: uuid.UUID,
        conversation_id: uuid.UUID,
        filename: str | None = None,
        content_type: str | None = None,
        size_bytes: int | None = None,
        storage_uri: str | None = None,
        status: str = "pending",
    ) -> Attachment:
        """Write an attachment record."""
        att = Attachment(
            attachment_id=attachment_id,
            conversation_id=conversation_id,
            filename=filename,
            content_type=content_type,
            size_bytes=size_bytes,
            storage_uri=storage_uri,
            status=status,
        )
        self.session.merge(att)
        return att

    def write_chunk(self, record: ChunkRecord) -> Chunk:
        """Write a single chunk record."""
        chunk = Chunk(
            tenant_id=record.tenant_id,
            chunk_id=record.chunk_id,
            conversation_id=record.conversation_id,
            attachment_id=record.attachment_id,
            is_attachment=record.is_attachment,
            is_summary=record.is_summary,
            chunk_type=record.chunk_type,
            text=record.text,
            embedding=record.embedding,
            position=record.position,
            char_start=record.char_start,
            char_end=record.char_end,
            section_path=record.section_path,
            extra_data=record.extra_data,
        )
        self.session.merge(chunk)
        return chunk

    def _write_conversation_and_attachments(
        self, tenant_id: str, results: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Write conversation and attachment records."""
        conv_data = results.get("conversation")
        if not conv_data:
            return None

        self.write_conversation(
            tenant_id=tenant_id,
            conversation_id=conv_data["conversation_id"],
            folder_name=conv_data.get("folder_name", ""),
            subject=conv_data.get("subject"),
            smart_subject=conv_data.get("smart_subject"),
            summary_text=conv_data.get("summary_text"),
            participants=conv_data.get("participants"),
            messages=conv_data.get("messages"),
            earliest_date=conv_data.get("earliest_date"),
            latest_date=conv_data.get("latest_date"),
            storage_uri=conv_data.get("storage_uri"),
            extra_data=conv_data.get("extra_data"),
        )

        for att_data in results.get("attachments", []):
            self.write_attachment(
                attachment_id=att_data["attachment_id"],
                conversation_id=att_data["conversation_id"],
                filename=att_data.get("filename"),
                content_type=att_data.get("content_type"),
                size_bytes=att_data.get("size_bytes"),
                storage_uri=att_data.get("storage_uri"),
                status=att_data.get("status", "pending"),
            )

        return conv_data

    def _synchronize_chunks(
        self, tenant_id: str, conv_id: uuid.UUID, chunks_data: list[dict[str, Any]]
    ) -> None:
        """Write new chunks and clean up stale ones."""
        current_chunk_ids = []
        for chunk_data in chunks_data:
            source = chunk_data.get("source") or (
                "attachment" if chunk_data.get("is_attachment") else "email_body"
            )
            chunk_data_with_meta = ensure_chunk_metadata(chunk_data, source=source)

            record = ChunkRecord(
                tenant_id=tenant_id,
                chunk_id=chunk_data_with_meta["chunk_id"],
                conversation_id=chunk_data_with_meta["conversation_id"],
                text=chunk_data_with_meta["text"],
                chunk_type=chunk_data_with_meta.get("chunk_type", "message_body"),
                is_summary=chunk_data_with_meta.get("is_summary", False),
                embedding=chunk_data_with_meta.get("embedding"),
                position=chunk_data_with_meta.get("position", 0),
                char_start=chunk_data_with_meta.get("char_start", 0),
                char_end=chunk_data_with_meta.get("char_end", 0),
                section_path=chunk_data_with_meta.get("section_path"),
                is_attachment=chunk_data_with_meta.get("is_attachment", False),
                attachment_id=chunk_data_with_meta.get("attachment_id"),
                extra_data=chunk_data_with_meta.get("extra_data"),
            )

            self.write_chunk(record)
            current_chunk_ids.append(record.chunk_id)

        # Cleanup stale chunks
        from sqlalchemy import delete

        stmt = delete(Chunk).where(Chunk.conversation_id == conv_id)
        if current_chunk_ids:
            stmt = stmt.where(Chunk.chunk_id.notin_(current_chunk_ids))

        deleted = self.session.execute(stmt)
        if deleted.rowcount > 0:
            logger.info(f"Cleaned up {deleted.rowcount} stale chunks for {conv_id}")

    def _synchronize_graph(
        self, tenant_id: str, conv_id: uuid.UUID | None, graph_data: dict[str, Any]
    ) -> None:
        """Upsert graph nodes and atomically update edges."""
        from cortex.db.models import EntityEdge, EntityNode
        from sqlalchemy import delete, select
        from sqlalchemy.dialects.postgresql import insert

        nodes = graph_data.get("nodes", [])
        edges = graph_data.get("edges", [])

        if not nodes and not edges:
            return

        # 1. Atomically upsert nodes using INSERT ... ON CONFLICT
        if nodes:
            node_values = [
                {
                    "tenant_id": tenant_id,
                    "name": n["name"],
                    "type": n["type"],
                    "description": f"Extracted from conversation {conv_id or 'unknown'}",
                    "properties": n.get("properties", {}),
                }
                for n in nodes
            ]

            # This statement inserts new nodes and does nothing if a node with the
            # same (tenant_id, name) already exists. It is atomic.
            upsert_stmt = insert(EntityNode).values(node_values)
            upsert_stmt = upsert_stmt.on_conflict_do_nothing(
                index_elements=["tenant_id", "name"]
            )
            self.session.execute(upsert_stmt)

        # 2. Fetch all required node IDs in a single query to build the map
        node_names = [n["name"] for n in nodes]
        stmt = select(EntityNode).where(
            EntityNode.tenant_id == tenant_id, EntityNode.name.in_(node_names)
        )
        existing_nodes = self.session.execute(stmt).scalars().all()
        node_map = {node.name: node.node_id for node in existing_nodes}

        # 3. Atomically update edges for the conversation (delete all, then insert all)
        if conv_id:
            self.session.execute(
                delete(EntityEdge).where(EntityEdge.conversation_id == conv_id)
            )

        new_edges = []
        for edge in edges:
            source_id = node_map.get(edge["source"])
            target_id = node_map.get(edge["target"])
            if source_id and target_id:
                new_edges.append(
                    EntityEdge(
                        tenant_id=tenant_id,
                        source_id=source_id,
                        target_id=target_id,
                        relation=edge["relation"],
                        description=edge["description"],
                        conversation_id=conv_id,
                    )
                )

        if new_edges:
            self.session.add_all(new_edges)

        logger.info(
            f"Synchronized graph for conversation {conv_id}: upserted {len(nodes)} nodes, set {len(new_edges)} edges."
        )

    def write_job_results(self, job: IngestJobRequest, results: dict[str, Any]) -> None:
        """
        Write all ingestion results to the database in a single transaction.
        """
        try:
            conv_data = self._write_conversation_and_attachments(job.tenant_id, results)

            if conv_data:
                conv_id = conv_data["conversation_id"]
                self._synchronize_chunks(
                    job.tenant_id, conv_id, results.get("chunks", [])
                )
                self._synchronize_graph(
                    job.tenant_id, conv_id, results.get("graph", {})
                )

            self.session.commit()
            logger.info(f"Successfully wrote job results for {job.job_id}")

        except Exception as e:
            self.session.rollback()
            logger.error(f"Failed to write job results: {e}")
            raise

    def write_chunks_with_dedup(
        self,
        chunks: list[dict[str, Any]],
        tenant_id: str,
        conversation_id: uuid.UUID,
        source: str = "email_body",
    ) -> tuple[int, int]:
        """
        Write chunks with deduplication based on content_hash.

        Returns:
            Tuple of (written_count, skipped_count)
        """
        if not chunks:
            return 0, 0

        # Ensure all chunks have metadata and get content hashes
        hashes = []
        for chunk_data in chunks:
            chunk_data = ensure_chunk_metadata(chunk_data, source=source)
            hashes.append(chunk_data["extra_data"]["content_hash"])

        # Performance: Bulk-fetch existing chunks to avoid N+1 query.
        from sqlalchemy import select

        stmt = select(Chunk.extra_data["content_hash"]).where(
            Chunk.conversation_id == conversation_id,
            Chunk.extra_data["content_hash"].astext.in_(hashes),
        )
        existing_hashes = {h for h, in self.session.execute(stmt)}

        written = 0
        for chunk_data in chunks:
            content_hash = chunk_data["extra_data"]["content_hash"]
            if content_hash in existing_hashes:
                logger.debug(f"Skipping duplicate chunk: {content_hash[:16]}...")
                continue

            record = ChunkRecord(
                tenant_id=tenant_id,
                chunk_id=chunk_data.get("chunk_id", uuid.uuid4()),
                conversation_id=conversation_id,
                text=chunk_data["text"],
                position=chunk_data.get("position", 0),
                char_start=chunk_data.get("char_start", 0),
                char_end=chunk_data.get("char_end", 0),
                section_path=chunk_data.get("section_path"),
                is_attachment=chunk_data.get("is_attachment", False),
                attachment_id=chunk_data.get("attachment_id"),
                extra_data=chunk_data.get("extra_data"),
            )
            self.write_chunk(record)
            written += 1

        skipped = len(chunks) - written
        self.session.commit()
        logger.info(f"Wrote {written} chunks, skipped {skipped} duplicates")
        return written, skipped
