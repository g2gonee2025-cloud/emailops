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
from typing import Any

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

    def write_job_results(self, job: IngestJobRequest, results: dict[str, Any]) -> None:
        """
        Write ingestion results to DB.

        Expected results format:
        {
            "conversation": {...},
            "attachments": [...],
            "chunks": [...]
        }
        """
        try:
            # 1. Write conversation
            conv_data = results.get("conversation", {})
            if conv_data:
                self.write_conversation(
                    conversation_id=conv_data["conversation_id"],
                    tenant_id=job.tenant_id,
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

            # 2. Write attachments
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

            # 3. Write chunks with metadata
            current_chunk_ids = []
            for chunk_data in results.get("chunks", []):
                source = chunk_data.get("source") or (
                    "attachment" if chunk_data.get("is_attachment") else "email_body"
                )
                chunk_data = ensure_chunk_metadata(chunk_data, source=source)

                record = ChunkRecord(
                    tenant_id=job.tenant_id,
                    chunk_id=chunk_data["chunk_id"],
                    conversation_id=chunk_data["conversation_id"],
                    text=chunk_data["text"],
                    chunk_type=chunk_data.get("chunk_type", "message_body"),
                    is_summary=chunk_data.get("is_summary", False),
                    embedding=chunk_data.get("embedding"),
                    position=chunk_data.get("position", 0),
                    char_start=chunk_data.get("char_start", 0),
                    char_end=chunk_data.get("char_end", 0),
                    section_path=chunk_data.get("section_path"),
                    is_attachment=chunk_data.get("is_attachment", False),
                    attachment_id=chunk_data.get("attachment_id"),
                    extra_data=chunk_data.get("extra_data"),
                )
                self.write_chunk(record)
                current_chunk_ids.append(chunk_data["chunk_id"])

            # 4. Write Knowledge Graph (Nodes & Edges)
            graph_data = results.get("graph", {})
            nodes = graph_data.get("nodes", [])
            edges = graph_data.get("edges", [])

            # Track conversation ID for edge context
            conv_id = conv_data.get("conversation_id") if conv_data else None

            if nodes or edges:
                from cortex.db.models import EntityEdge, EntityNode
                from sqlalchemy import select

                node_map = {}  # name -> db_id

                # Upsert Nodes (Scoped to Tenant)
                for node_dict in nodes:
                    name = node_dict["name"]
                    stmt = select(EntityNode).where(
                        EntityNode.tenant_id == job.tenant_id, EntityNode.name == name
                    )
                    existing_node = self.session.execute(stmt).scalar_one_or_none()

                    if not existing_node:
                        new_node = EntityNode(
                            tenant_id=job.tenant_id,
                            name=name,
                            type=node_dict["type"],
                            description=f"Extracted from conversation {conv_id or 'unknown'}",
                            properties=node_dict.get("properties", {}),
                        )
                        self.session.add(new_node)
                        self.session.flush()  # flush to get ID
                        node_map[name] = new_node.node_id
                    else:
                        node_map[name] = existing_node.node_id

                # Write Edges (Scoped to Conversation)
                for edge_dict in edges:
                    src_id = node_map.get(edge_dict["source"])
                    target_id = node_map.get(edge_dict["target"])

                    if src_id and target_id:
                        edge = EntityEdge(
                            tenant_id=job.tenant_id,
                            source_id=src_id,
                            target_id=target_id,
                            relation=edge_dict["relation"],
                            description=edge_dict["description"],
                            conversation_id=conv_id,
                        )
                        self.session.merge(edge)
                        # Note: We don't have stable IDs for edges yet, so this merge implies
                        # we rely on DB definition of PK. Assuming auto-increment or similar.
                        # For now, we will just add them.
                        # self.session.add(edge) # merge matches on PK

                logger.info(f"Wrote {len(nodes)} nodes and {len(edges)} edges to graph")

            # 5. Cleanup stale chunks
            # If we processed the conversation, any existing chunks NOT in the new list should be removed.
            # This handles cases where content changed (new hash -> new ID) or attachments were deleted.
            if conv_data and "conversation_id" in conv_data:
                cid = conv_data["conversation_id"]
                from sqlalchemy import delete

                # Delete chunks for this conversation that are NOT in the current set
                if current_chunk_ids:
                    stmt = delete(Chunk).where(
                        Chunk.conversation_id == cid,
                        Chunk.chunk_id.notin_(current_chunk_ids),
                    )
                else:
                    # If no chunks produced (empty convo?), delete all existing chunks
                    stmt = delete(Chunk).where(Chunk.conversation_id == cid)

                deleted = self.session.execute(stmt)
                logger.info(f"Cleaned up {deleted.rowcount} stale chunks for {cid}")

                # Cleanup stale graph edges for this conversation?
                # Ideally yes, but current Edge model doesn't make it easy without identifying 'stale'.
                # For this iteration, we accept that edges accumulate or we wipe edges for this convo first.
                # Wiping edges for this conversation is safer for atomicity.
                # stmt_edges = delete(EntityEdge).where(EntityEdge.conversation_id == cid)
                # self.session.execute(stmt_edges)
                # But we just added new ones!
                # Correct logic: Wipe FIRST, then Add.

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
        written = 0
        skipped = 0

        for chunk_data in chunks:
            chunk_data = ensure_chunk_metadata(chunk_data, source=source)
            content_hash = chunk_data["extra_data"]["content_hash"]

            # Check for existing chunk with same hash
            existing = (
                self.session.query(Chunk)
                .filter(Chunk.conversation_id == conversation_id)
                .filter(Chunk.extra_data["content_hash"].astext == content_hash)
                .first()
            )

            if existing:
                logger.debug(f"Skipping duplicate chunk: {content_hash[:16]}...")
                skipped += 1
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

        self.session.commit()
        logger.info(f"Wrote {written} chunks, skipped {skipped} duplicates")
        return written, skipped
