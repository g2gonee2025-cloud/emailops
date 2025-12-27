"""
Database models for Cortex.

Implements simplified schema per §4.1 of the Canonical Blueprint.
Tables: Conversation, Attachment, Chunk, AuditLog
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from pgvector.sqlalchemy import HALFVEC
from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

# SQLAlchemy cascade constant to avoid duplication (S1192)
CASCADE_DELETE_ORPHAN = "all, delete-orphan"
FK_CONVERSATION_ID = "conversations.conversation_id"


class Base(DeclarativeBase):
    """Base class for all models."""

    pass


class Conversation(Base):
    """
    Represents a conversation (email thread) from a single S3 folder.

    Each folder = one conversation with its manifest.json.
    """

    __tablename__ = "conversations"

    conversation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    tenant_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    folder_name: Mapped[str] = mapped_column(String(512), nullable=False)
    subject: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    smart_subject: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    participants: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSONB, nullable=True
    )  # [{name, smtp}, ...]
    messages: Mapped[Optional[List[Dict[str, Any]]]] = mapped_column(
        JSONB, nullable=True
    )  # Full messages array from manifest
    earliest_date: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    latest_date: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    storage_uri: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    extra_data: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB, nullable=True)
    summary_text: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True, comment="AI-generated summary of the conversation"
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    # Relationships
    attachments: Mapped[List["Attachment"]] = relationship(
        back_populates="conversation", cascade=CASCADE_DELETE_ORPHAN
    )
    chunks: Mapped[List["Chunk"]] = relationship(
        back_populates="conversation", cascade=CASCADE_DELETE_ORPHAN
    )

    __table_args__ = (
        Index("ix_conversations_tenant_folder", "tenant_id", "folder_name"),
        Index(
            "ix_conversations_participants_gin",
            "participants",
            postgresql_using="gin",
        ),
    )

    def __repr__(self) -> str:
        """
        Return a string representation of the Conversation, redacting PII.
        """
        return (
            f"<Conversation(conversation_id='{self.conversation_id}', "
            f"tenant_id='{self.tenant_id}', "
            f"folder_name='{self.folder_name}', "
            f"subject='[REDACTED]', "
            f"smart_subject='[REDACTED]', "
            f"participants='[REDACTED]')>"
        )


class Attachment(Base):
    """
    Represents an attachment from a conversation.
    """

    __tablename__ = "attachments"

    attachment_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey(FK_CONVERSATION_ID, ondelete="CASCADE"),
        nullable=False,
    )
    filename: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    content_type: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    size_bytes: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    storage_uri: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(
        String(32), nullable=False, default="pending"
    )  # pending|parsed|failed

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Relationships
    conversation: Mapped["Conversation"] = relationship(back_populates="attachments")
    chunks: Mapped[List["Chunk"]] = relationship(
        back_populates="attachment", cascade=CASCADE_DELETE_ORPHAN
    )

    __table_args__ = (Index("ix_attachments_conversation", "conversation_id"),)


class Chunk(Base):
    """
    Represents a text chunk for RAG retrieval.

    Can be from conversation body or attachment content.
    """

    __tablename__ = "chunks"

    chunk_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    # Denormalized tenant for efficient filtering in retrieval paths
    tenant_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey(FK_CONVERSATION_ID, ondelete="CASCADE"),
        nullable=False,
    )
    attachment_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("attachments.attachment_id", ondelete="CASCADE"),
        nullable=True,
    )
    is_attachment: Mapped[bool] = mapped_column(Boolean, nullable=False)
    is_summary: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    chunk_type: Mapped[str] = mapped_column(
        String(32), nullable=False, default="message_body"
    )
    text: Mapped[str] = mapped_column(Text, nullable=False)
    tsv_text: Mapped[Optional[Any]] = mapped_column(TSVECTOR, nullable=True)
    embedding = mapped_column(
        HALFVEC(3840), nullable=True
    )  # pgvector, KaLM full dim (float16)
    position: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    char_start: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    char_end: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    section_path: Mapped[Optional[str]] = mapped_column(
        String(256), nullable=True
    )  # e.g., "email:body", "attachment:report.pdf"
    extra_data: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Relationships
    conversation: Mapped["Conversation"] = relationship(back_populates="chunks")
    attachment: Mapped[Optional["Attachment"]] = relationship(back_populates="chunks")

    __table_args__ = (
        Index("ix_chunks_conversation", "conversation_id"),
        Index("ix_chunks_is_attachment", "is_attachment"),
        # FTS Index
        Index("ix_chunks_tsv_text", "tsv_text", postgresql_using="gin"),
        Index("ix_chunks_extra_data_gin", "extra_data", postgresql_using="gin"),
        # Vector Index (HNSW for high-dim halfvec)
        # Note: Created via migration 011
    )


class AuditLog(Base):
    """
    Audit log for tracking actions.
    """

    __tablename__ = "audit_log"

    audit_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    ts: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False, index=True
    )
    tenant_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    user_or_agent: Mapped[str] = mapped_column(String(256), nullable=False)
    action: Mapped[str] = mapped_column(String(128), nullable=False)
    input_hash: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    output_hash: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    policy_decisions: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSONB, nullable=True
    )
    risk_level: Mapped[str] = mapped_column(
        String(32), server_default="low", nullable=False
    )
    metadata: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB, nullable=True)


class EntityNode(Base):
    """
    Graph RAG: Represents a distinct entity (Person, Project, Org, etc.).
    Nodes are global per tenant.
    """

    __tablename__ = "entity_nodes"

    node_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    tenant_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(512), nullable=False)
    type: Mapped[str] = mapped_column(String(64), nullable=False)  # e.g., "PERSON"
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    properties: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    # Relationships
    outgoing_edges: Mapped[List["EntityEdge"]] = relationship(
        "EntityEdge",
        foreign_keys="[EntityEdge.source_id]",
        back_populates="source_node",
        cascade=CASCADE_DELETE_ORPHAN,
    )
    incoming_edges: Mapped[List["EntityEdge"]] = relationship(
        "EntityEdge",
        foreign_keys="[EntityEdge.target_id]",
        back_populates="target_node",
        cascade=CASCADE_DELETE_ORPHAN,
    )

    __table_args__ = (
        Index("ix_entity_nodes_tenant_name", "tenant_id", "name", unique=True),
    )


class EntityEdge(Base):
    """
    Graph RAG: Represents a relationship between two entities.
    Edges are derived from specific conversations.
    """

    __tablename__ = "entity_edges"

    edge_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    tenant_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    source_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("entity_nodes.node_id", ondelete="CASCADE"),
        nullable=False,
    )
    target_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("entity_nodes.node_id", ondelete="CASCADE"),
        nullable=False,
    )
    relation: Mapped[str] = mapped_column(
        String(128), nullable=False
    )  # e.g., "MANAGED_BY"
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    weight: Mapped[float] = mapped_column(
        "weight", nullable=False, default=1.0
    )  # Confidence score

    # Provenance
    conversation_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey(FK_CONVERSATION_ID, ondelete="SET NULL"),
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Relationships
    source_node: Mapped["EntityNode"] = relationship(
        "EntityNode", foreign_keys=[source_id], back_populates="outgoing_edges"
    )
    target_node: Mapped["EntityNode"] = relationship(
        "EntityNode", foreign_keys=[target_id], back_populates="incoming_edges"
    )

    __table_args__ = (
        Index("ix_entity_edges_source", "source_id"),
        Index("ix_entity_edges_target", "target_id"),
        Index("ix_entity_edges_conversation", "conversation_id"),
    )
