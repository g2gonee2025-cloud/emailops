"""
SQLAlchemy models for Cortex (EmailOps Edition).

Implements the schema defined in §4.1 of the Canonical Blueprint.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, List, Optional

try:
    from pgvector.sqlalchemy import Vector
except ImportError:
    # Fallback for environments without pgvector installed
    from sqlalchemy.types import UserDefinedType

    class Vector(UserDefinedType):
        def __init__(self, dim):
            self.dim = dim

        def get_col_spec(self, **kw):
            return "vector(%d)" % self.dim

        def bind_processor(self, dialect):
            def process(value):
                return value
            return process

        def result_processor(self, dialect, coltype):
            def process(value):
                return value
            return process
from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, TSVECTOR, UUID
try:
    from sqlalchemy.orm import DeclarativeBase
except ImportError:
    # Fallback for older SQLAlchemy versions
    from sqlalchemy.ext.declarative import declarative_base
    DeclarativeBase = declarative_base()

try:
    from sqlalchemy.orm import Mapped, mapped_column
except ImportError:
    # Fallback for older SQLAlchemy versions
    from sqlalchemy import Column
    from sqlalchemy.orm import relationship
    
    # Mock Mapped for type hinting compatibility
    class Mapped:
        def __class_getitem__(cls, item):
            return item

    def mapped_column(*args, **kwargs):
        return Column(*args, **kwargs)

from sqlalchemy.orm import relationship

from cortex.config.loader import get_config

# Load config to get embedding dimension
_config = get_config()
EMBEDDING_DIM = _config.embedding.output_dimensionality


if isinstance(DeclarativeBase, type):
    class Base(DeclarativeBase):
        """Base class for all models."""
        __abstract__ = True
else:
    Base = DeclarativeBase


class Thread(Base):
    """
    Represents an email thread (conversation).
    
    Blueprint §4.1:
    * thread_id (uuid, pk)
    * tenant_id (text, not null)
    * subject_norm (text)
    * original_subject (text)
    * created_at (timestamptz, not null, UTC)
    * updated_at (timestamptz, not null, UTC)
    * metadata (jsonb)
    """
    __tablename__ = "threads"

    thread_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )
    tenant_id: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    subject_norm: Mapped[Optional[str]] = mapped_column(Text)
    original_subject: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    metadata_: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSONB, default=dict
    )

    # Relationships
    messages: Mapped[List["Message"]] = relationship(
        "Message", back_populates="thread", cascade="all, delete-orphan"
    )
    chunks: Mapped[List["Chunk"]] = relationship(
        "Chunk", back_populates="thread", cascade="all, delete-orphan"
    )


class Message(Base):
    """
    Represents an individual email message.
    
    Blueprint §4.1:
    * message_id (text, pk)
    * thread_id (uuid, fk -> threads.thread_id, not null)
    * folder (text)
    * sent_at (timestamptz, nullable, UTC)
    * recv_at (timestamptz, nullable, UTC)
    * from_addr (text, not null)
    * to_addrs (text[])
    * cc_addrs (text[])
    * bcc_addrs (text[])
    * subject (text)
    * body_plain (text)
    * body_html (text)
    * has_quoted_mask (boolean, default false)
    * raw_storage_uri (text)
    * tenant_id (text, not null)
    * tsv_subject_body (tsvector, indexed)
    * metadata (jsonb)
    """
    __tablename__ = "messages"

    message_id: Mapped[str] = mapped_column(Text, primary_key=True)
    thread_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("threads.thread_id", ondelete="RESTRICT"),
        nullable=False,
        index=True
    )
    folder: Mapped[Optional[str]] = mapped_column(Text)
    sent_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    recv_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    from_addr: Mapped[str] = mapped_column(Text, nullable=False)
    to_addrs: Mapped[List[str]] = mapped_column(ARRAY(Text), default=list)
    cc_addrs: Mapped[List[str]] = mapped_column(ARRAY(Text), default=list)
    bcc_addrs: Mapped[List[str]] = mapped_column(ARRAY(Text), default=list)
    subject: Mapped[Optional[str]] = mapped_column(Text)
    body_plain: Mapped[Optional[str]] = mapped_column(Text)
    body_html: Mapped[Optional[str]] = mapped_column(Text)
    has_quoted_mask: Mapped[bool] = mapped_column(Boolean, default=False)
    raw_storage_uri: Mapped[Optional[str]] = mapped_column(Text)
    tenant_id: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    tsv_subject_body: Mapped[Optional[Any]] = mapped_column(TSVECTOR)
    metadata_: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSONB, default=dict
    )

    # Relationships
    thread: Mapped["Thread"] = relationship("Thread", back_populates="messages")
    attachments: Mapped[List["Attachment"]] = relationship(
        "Attachment", back_populates="message", cascade="all, delete-orphan"
    )
    chunks: Mapped[List["Chunk"]] = relationship(
        "Chunk", back_populates="message", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("ix_messages_tsv_subject_body", "tsv_subject_body", postgresql_using="gin"),
    )


class Attachment(Base):
    """
    Represents an email attachment.
    
    Blueprint §4.1:
    * attachment_id (uuid, pk)
    * message_id (text, fk -> messages.message_id, not null)
    * filename (text)
    * mime_type (text)
    * storage_uri_raw (text)
    * storage_uri_extracted (text)
    * status (enum: pending|parsed|unparsed_password_protected|failed)
    * extracted_chars (int)
    * tenant_id (text, not null)
    * metadata (jsonb)
    """
    __tablename__ = "attachments"

    attachment_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )
    message_id: Mapped[str] = mapped_column(
        ForeignKey("messages.message_id", ondelete="RESTRICT"),
        nullable=False,
        index=True
    )
    filename: Mapped[Optional[str]] = mapped_column(Text)
    mime_type: Mapped[Optional[str]] = mapped_column(Text)
    storage_uri_raw: Mapped[Optional[str]] = mapped_column(Text)
    storage_uri_extracted: Mapped[Optional[str]] = mapped_column(Text)
    status: Mapped[str] = mapped_column(Text, default="pending")
    extracted_chars: Mapped[int] = mapped_column(Integer, default=0)
    tenant_id: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    metadata_: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSONB, default=dict
    )

    # Relationships
    message: Mapped["Message"] = relationship("Message", back_populates="attachments")
    chunks: Mapped[List["Chunk"]] = relationship(
        "Chunk", back_populates="attachment", cascade="all, delete-orphan"
    )


class Chunk(Base):
    """
    Represents a searchable chunk of text.
    
    Blueprint §4.1:
    * chunk_id (uuid, pk)
    * thread_id (uuid, fk -> threads.thread_id, not null)
    * message_id (text, fk -> messages.message_id, nullable)
    * attachment_id (uuid, fk -> attachments.attachment_id, nullable)
    * chunk_type (enum: message_body|attachment_text|attachment_table|quoted_history|other)
    * text (text)
    * summary (text)
    * section_path (text)
    * position (int)
    * char_start (int)
    * char_end (int)
    * embedding (vector)
    * embedding_model (text)
    * tenant_id (text, not null)
    * tsv_text (tsvector, indexed)
    * metadata (jsonb)
    """
    __tablename__ = "chunks"

    chunk_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )
    thread_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("threads.thread_id", ondelete="RESTRICT"),
        nullable=False,
        index=True
    )
    message_id: Mapped[Optional[str]] = mapped_column(
        ForeignKey("messages.message_id", ondelete="RESTRICT"),
        nullable=True,
        index=True
    )
    attachment_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        ForeignKey("attachments.attachment_id", ondelete="RESTRICT"),
        nullable=True,
        index=True
    )
    chunk_type: Mapped[str] = mapped_column(Text, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    summary: Mapped[Optional[str]] = mapped_column(Text)
    section_path: Mapped[str] = mapped_column(Text, nullable=False)
    position: Mapped[int] = mapped_column(Integer, nullable=False)
    char_start: Mapped[int] = mapped_column(Integer, nullable=False)
    char_end: Mapped[int] = mapped_column(Integer, nullable=False)
    embedding: Mapped[Optional[Any]] = mapped_column(Vector(EMBEDDING_DIM))
    embedding_model: Mapped[Optional[str]] = mapped_column(Text)
    tenant_id: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    tsv_text: Mapped[Optional[Any]] = mapped_column(TSVECTOR)
    metadata_: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSONB, default=dict
    )

    # Relationships
    thread: Mapped["Thread"] = relationship("Thread", back_populates="chunks")
    message: Mapped[Optional["Message"]] = relationship("Message", back_populates="chunks")
    attachment: Mapped[Optional["Attachment"]] = relationship("Attachment", back_populates="chunks")

    __table_args__ = (
        Index("ix_chunks_tsv_text", "tsv_text", postgresql_using="gin"),
        Index("ix_chunks_embedding", "embedding", postgresql_using="hnsw", postgresql_with={"m": 16, "ef_construction": 64}, postgresql_ops={"embedding": "vector_cosine_ops"}),
    )

    # -------------------------------------------------------------------------
    # Metadata Property Helpers (Blueprint §4.1)
    # -------------------------------------------------------------------------
    
    @property
    def content_hash(self) -> Optional[str]:
        """Get content_hash from metadata (used for dedup)."""
        return self.metadata_.get("content_hash") if self.metadata_ else None
    
    @content_hash.setter
    def content_hash(self, value: str) -> None:
        """Set content_hash in metadata."""
        if self.metadata_ is None:
            self.metadata_ = {}
        self.metadata_["content_hash"] = value
    
    @property
    def pre_cleaned(self) -> Optional[str]:
        """Get pre_cleaned text from metadata."""
        return self.metadata_.get("pre_cleaned") if self.metadata_ else None
    
    @pre_cleaned.setter
    def pre_cleaned(self, value: str) -> None:
        """Set pre_cleaned in metadata."""
        if self.metadata_ is None:
            self.metadata_ = {}
        self.metadata_["pre_cleaned"] = value
    
    @property
    def cleaning_version(self) -> Optional[str]:
        """Get cleaning_version from metadata."""
        return self.metadata_.get("cleaning_version") if self.metadata_ else None
    
    @cleaning_version.setter
    def cleaning_version(self, value: str) -> None:
        """Set cleaning_version in metadata."""
        if self.metadata_ is None:
            self.metadata_ = {}
        self.metadata_["cleaning_version"] = value
    
    @property
    def source(self) -> Optional[str]:
        """Get source type from metadata (email|attachment|ocr)."""
        return self.metadata_.get("source") if self.metadata_ else None
    
    @source.setter
    def source(self, value: str) -> None:
        """Set source in metadata."""
        if self.metadata_ is None:
            self.metadata_ = {}
        self.metadata_["source"] = value


class AuditLog(Base):
    """
    Audit log for security and compliance.
    
    Blueprint §4.1:
    * audit_id (uuid, pk)
    * ts (timestamptz, not null, UTC)
    * tenant_id (text, not null)
    * user_or_agent (text, not null)
    * action (text, not null)
    * input_hash (text)
    * output_hash (text)
    * policy_decisions (jsonb)
    * risk_level (enum: low|medium|high)
    * metadata (jsonb)
    """
    __tablename__ = "audit_log"

    audit_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )
    ts: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=func.now()
    )
    tenant_id: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    user_or_agent: Mapped[str] = mapped_column(Text, nullable=False)
    action: Mapped[str] = mapped_column(Text, nullable=False)
    input_hash: Mapped[Optional[str]] = mapped_column(Text)
    output_hash: Mapped[Optional[str]] = mapped_column(Text)
    policy_decisions: Mapped[Optional[dict[str, Any]]] = mapped_column(JSONB)
    risk_level: Mapped[str] = mapped_column(Text, default="low")
    metadata_: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSONB, default=dict
    )


class IngestJob(Base):
    """
    Tracks ingestion jobs.
    
    Blueprint §4.1:
    * job_id (text, pk)
    * tenant_id (text, not null)
    * source_type (text, not null)
    * status (enum: pending|processing|completed|failed)
    * created_at (timestamptz, not null, UTC)
    * updated_at (timestamptz, not null, UTC)
    * stats (jsonb)
    * error_message (text)
    * metadata (jsonb)
    """
    __tablename__ = "ingest_jobs"

    job_id: Mapped[str] = mapped_column(Text, primary_key=True)
    tenant_id: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    source_type: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(Text, default="pending", index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=func.now(), onupdate=func.now()
    )
    stats: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    error_message: Mapped[Optional[str]] = mapped_column(Text)
    metadata_: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSONB, default=dict
    )


class FactsLedger(Base):
    """
    Persisted facts ledger for thread summarization.
    
    Blueprint §10.4.1:
    * ledger_id (uuid, pk)
    * thread_id (uuid, fk -> threads.thread_id, unique)
    * facts (jsonb) - List of Fact objects
    * entities (jsonb) - List of Entity objects
    * decisions (jsonb) - List of Decision objects
    * action_items (jsonb) - List of ActionItem objects
    * unresolved_questions (jsonb) - List of UnresolvedQuestion objects
    * created_at (timestamptz, not null, UTC)
    * updated_at (timestamptz, not null, UTC)
    * tenant_id (text, not null)
    """
    __tablename__ = "facts_ledger"

    ledger_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )
    thread_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("threads.thread_id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True
    )
    facts: Mapped[List[dict[str, Any]]] = mapped_column(JSONB, default=list)
    entities: Mapped[List[dict[str, Any]]] = mapped_column(JSONB, default=list)
    decisions: Mapped[List[dict[str, Any]]] = mapped_column(JSONB, default=list)
    action_items: Mapped[List[dict[str, Any]]] = mapped_column(JSONB, default=list)
    unresolved_questions: Mapped[List[dict[str, Any]]] = mapped_column(JSONB, default=list)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=func.now(), onupdate=func.now()
    )
    tenant_id: Mapped[str] = mapped_column(Text, nullable=False, index=True)

    # Relationships
    thread: Mapped["Thread"] = relationship("Thread")
