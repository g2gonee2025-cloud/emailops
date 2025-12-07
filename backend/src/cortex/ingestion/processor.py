"""
Ingestion Processor for EmailOps.

Processes email conversations from S3, generates embeddings via Vertex AI,
and stores them in PostgreSQL with pgvector.

Implements §6 and §7 of the Canonical Blueprint.
"""
from __future__ import annotations

import hashlib
import json
import logging
import mimetypes
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from cortex.config.loader import get_config
from cortex.db.models import Attachment, Chunk, IngestJob, Message, Thread
from cortex.embeddings.client import EmbeddingsClient
from cortex.ingestion.parser_email import normalize_subject
from cortex.ingestion.s3_source import S3ConversationFolder, S3SourceHandler
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

logger = logging.getLogger(__name__)

# Chunk size limits per Blueprint §7.1
MAX_CHUNK_CHARS = 2000
CHUNK_OVERLAP = 200


@dataclass
class ConversationData:
    """Parsed conversation data from S3."""

    folder_name: str
    manifest: Dict[str, Any]
    body_text: str
    attachments: List[Dict[str, Any]] = field(default_factory=list)
    subject: str = ""
    from_addr: str = ""
    to_addrs: List[str] = field(default_factory=list)
    sent_at: Optional[datetime] = None


@dataclass
class ProcessingStats:
    """Statistics for a processing run."""

    folders_processed: int = 0
    threads_created: int = 0
    messages_created: int = 0
    chunks_created: int = 0
    embeddings_generated: int = 0
    errors: int = 0
    skipped: int = 0


class IngestionProcessor:
    """
    Processes email conversations from S3 into the vector database.

    Blueprint §6: Ingestion Pipeline
    - Reads from S3 raw bucket
    - Parses email content
    - Chunks text
    - Generates embeddings
    - Stores in PostgreSQL with pgvector
    """

    def __init__(
        self,
        tenant_id: str = "default",
        batch_size: int = 10,
        db_url: Optional[str] = None,
    ):
        self.config = get_config()
        self.tenant_id = tenant_id
        self.batch_size = batch_size

        # Database connection
        self.db_url = db_url or self.config.database.url
        self.engine = create_engine(self.db_url)
        self.SessionLocal = sessionmaker(bind=self.engine)

        # S3 handler
        self.s3_handler = S3SourceHandler()

        # Embedding client
        self.embedder = EmbeddingsClient()

        # Processing stats
        self.stats = ProcessingStats()

    def process_folder(self, folder: S3ConversationFolder) -> Optional[str]:
        """
        Process a single conversation folder from S3.

        Returns:
            thread_id if successful, None if failed
        """
        try:
            # Stream conversation data directly from S3
            data = self.s3_handler.stream_conversation_data(folder)

            if not data.get("conversation_txt") and not data.get("manifest"):
                logger.warning(f"Empty folder: {folder.name}")
                self.stats.skipped += 1
                return None

            # Parse conversation data
            conv_data = self._parse_streamed_conversation(folder.name, data)
            if not conv_data:
                logger.warning(f"Failed to parse conversation: {folder.name}")
                self.stats.errors += 1
                return None

            # Store in database
            thread_id = self._store_conversation(conv_data)

            self.stats.folders_processed += 1
            return thread_id

        except Exception as e:
            logger.error(f"Error processing folder {folder.name}: {e}")
            self.stats.errors += 1
            return None

    def _parse_streamed_conversation(
        self, folder_name: str, data: Dict[str, Any]
    ) -> Optional[ConversationData]:
        """Parse streamed conversation data from S3."""
        try:
            manifest = data.get("manifest", {})
            body_text = data.get("conversation_txt", "")

            # Parse metadata from manifest
            subject = manifest.get("subject", folder_name)
            from_addr = manifest.get(
                "from", manifest.get("sender", "unknown@unknown.com")
            )
            to_addrs = manifest.get("to", [])
            if isinstance(to_addrs, str):
                to_addrs = [to_addrs]

            # Parse sent date
            sent_at = None
            if manifest.get("date"):
                try:
                    from dateutil.parser import parse as parse_date

                    sent_at = parse_date(manifest["date"])
                    if sent_at.tzinfo is None:
                        sent_at = sent_at.replace(tzinfo=timezone.utc)
                except Exception:
                    pass

            # Parse attachments
            attachments = data.get("attachments", [])

            return ConversationData(
                folder_name=folder_name,
                manifest=manifest,
                body_text=body_text,
                attachments=attachments,
                subject=subject,
                from_addr=from_addr,
                to_addrs=to_addrs,
                sent_at=sent_at,
            )

        except Exception as e:
            logger.error(f"Error parsing conversation {folder_name}: {e}")
            return None

    def _parse_conversation(
        self, folder_name: str, files: Dict[str, bytes]
    ) -> Optional[ConversationData]:
        """Parse downloaded files into ConversationData."""
        try:
            # Parse manifest.json
            manifest = {}
            if "manifest.json" in files:
                try:
                    content = files["manifest.json"].decode("utf-8-sig")
                    # Fix malformed JSON from Outlook export (triple quotes issue)
                    content = content.replace(':"""', ':""').replace(': """', ': ""')
                    manifest = json.loads(content)
                except Exception as e:
                    logger.warning(
                        f"Failed to parse manifest.json in {folder_name}: {e}"
                    )

            # Get body text
            body_text = ""
            for fname in ["body.txt", "body_plain.txt", "content.txt"]:
                if fname in files:
                    body_text = files[fname].decode("utf-8-sig", errors="replace")
                    break

            # If no body file, try to extract from manifest
            if not body_text and manifest.get("body"):
                body_text = manifest["body"]

            # Parse metadata from manifest
            subject = manifest.get("subject", folder_name)
            from_addr = manifest.get(
                "from", manifest.get("sender", "unknown@unknown.com")
            )
            to_addrs = manifest.get("to", [])
            if isinstance(to_addrs, str):
                to_addrs = [to_addrs]

            # Parse sent date
            sent_at = None
            if manifest.get("date"):
                try:
                    from dateutil.parser import parse as parse_date

                    sent_at = parse_date(manifest["date"])
                    if sent_at.tzinfo is None:
                        sent_at = sent_at.replace(tzinfo=timezone.utc)
                except Exception:
                    pass

            # Parse attachments
            attachments = []
            if manifest.get("attachments"):
                for att in manifest["attachments"]:
                    att_name = att.get("filename", att.get("name", "unknown"))
                    att_content = files.get(f"attachments/{att_name}", b"")
                    attachments.append(
                        {
                            "filename": att_name,
                            "mime_type": att.get(
                                "mime_type", "application/octet-stream"
                            ),
                            "content": att_content,
                            "size": len(att_content),
                        }
                    )

            return ConversationData(
                folder_name=folder_name,
                manifest=manifest,
                body_text=body_text,
                attachments=attachments,
                subject=subject,
                from_addr=from_addr,
                to_addrs=to_addrs,
                sent_at=sent_at,
            )

        except Exception as e:
            logger.error(f"Error parsing conversation {folder_name}: {e}")
            return None

    def _store_conversation(self, conv_data: ConversationData) -> str:
        """Store conversation in database with embeddings."""
        with self.SessionLocal() as session:
            try:
                # Create thread
                now = datetime.now(timezone.utc)
                thread_id = uuid.uuid4()

                # Normalize subject using canonical helper
                subject_norm = normalize_subject(conv_data.subject)

                thread = Thread(
                    thread_id=thread_id,
                    tenant_id=self.tenant_id,
                    subject_norm=subject_norm,
                    original_subject=conv_data.subject,
                    created_at=conv_data.sent_at or now,
                    updated_at=now,
                    metadata_={
                        "source": "s3_ingest",
                        "folder_name": conv_data.folder_name,
                        "manifest": conv_data.manifest,
                    },
                )
                session.add(thread)
                self.stats.threads_created += 1

                # Create message
                message_id = f"msg-{thread_id}-0"
                message = Message(
                    message_id=message_id,
                    thread_id=thread_id,
                    folder=conv_data.folder_name,
                    sent_at=conv_data.sent_at,
                    recv_at=conv_data.sent_at,
                    from_addr=conv_data.from_addr,
                    to_addrs=conv_data.to_addrs,
                    subject=conv_data.subject,
                    body_plain=conv_data.body_text[:50000]
                    if conv_data.body_text
                    else None,  # Limit size
                    tenant_id=self.tenant_id,
                    raw_storage_uri=f"s3://{self.config.storage.bucket_raw}/raw/outlook/{conv_data.folder_name}/",
                    metadata_={"source": "s3_ingest"},
                )
                session.add(message)
                self.stats.messages_created += 1

                # Create chunks with embeddings
                chunks = self._create_chunks(
                    session,
                    thread_id,
                    message_id,
                    conv_data.body_text,
                    conv_data.subject,
                )

                for chunk in chunks:
                    session.add(chunk)
                    self.stats.chunks_created += 1

                # Create attachments and their chunks
                for att_data in conv_data.attachments:
                    raw_filename = att_data.get("filename")
                    derived_name = raw_filename
                    if not derived_name:
                        candidate = att_data.get("path") or att_data.get("s3_key")
                        if candidate:
                            derived_name = Path(candidate).name
                    filename = derived_name or "attachment"

                    guessed_mime = mimetypes.guess_type(filename)[0]
                    mime_type = (
                        att_data.get("mime_type")
                        or guessed_mime
                        or "application/octet-stream"
                    )

                    content = att_data.get("content")
                    text_payload = att_data.get("text")
                    size = att_data.get("size")
                    if size is None:
                        if content is not None:
                            size = len(content)
                        elif text_payload:
                            size = len(text_payload)
                        else:
                            size = 0

                    status = "parsed" if (content or text_payload) else "pending"
                    storage_uri = att_data.get("storage_uri_raw")
                    if not storage_uri:
                        s3_key = att_data.get("s3_key")
                        if s3_key:
                            storage_uri = (
                                f"s3://{self.config.storage.bucket_raw}/{s3_key}"
                            )

                    metadata_source = att_data.get("metadata") or {}
                    metadata = dict(metadata_source)
                    metadata.setdefault("source", "s3_ingest")
                    if att_data.get("s3_key"):
                        metadata.setdefault("s3_key", att_data["s3_key"])

                    attachment = Attachment(
                        message_id=message_id,
                        filename=filename,
                        mime_type=mime_type,
                        storage_uri_raw=storage_uri,
                        status=status,
                        extracted_chars=size or 0,
                        tenant_id=self.tenant_id,
                        metadata_=metadata,
                    )
                    session.add(attachment)

                session.commit()
                logger.info(f"Stored thread {thread_id} with {len(chunks)} chunks")
                return str(thread_id)

            except Exception as e:
                session.rollback()
                logger.error(f"Error storing conversation: {e}")
                raise

    def _create_chunks(
        self,
        session: Session,
        thread_id: uuid.UUID,
        message_id: str,
        text: str,
        subject: str,
    ) -> List[Chunk]:
        """Create chunks with embeddings from text."""
        if not text or not text.strip():
            return []

        # Prepend subject to text for better context
        full_text = f"Subject: {subject}\n\n{text}" if subject else text

        # Split into chunks
        text_chunks = self._split_text(full_text)
        if not text_chunks:
            return []

        # Generate embeddings in batch
        try:
            embeddings = self.embedder.embed_batch(text_chunks)
            self.stats.embeddings_generated += len(embeddings)
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            # Return chunks without embeddings
            embeddings = [None] * len(text_chunks)

        chunks = []
        for i, (chunk_text, embedding) in enumerate(zip(text_chunks, embeddings)):
            content_hash = hashlib.sha256(chunk_text.encode()).hexdigest()[:16]

            chunk = Chunk(
                thread_id=thread_id,
                message_id=message_id,
                chunk_type="message_body",
                text=chunk_text,
                section_path=f"/message/{i}",
                position=i,
                char_start=i * (MAX_CHUNK_CHARS - CHUNK_OVERLAP),
                char_end=min((i + 1) * MAX_CHUNK_CHARS, len(full_text)),
                embedding=embedding,
                embedding_model=self.config.embedding.model_name,
                tenant_id=self.tenant_id,
                metadata_={"content_hash": content_hash, "source": "email"},
            )
            chunks.append(chunk)

        return chunks

    def _split_text(self, text: str) -> List[str]:
        """Split text into chunks with overlap."""
        if len(text) <= MAX_CHUNK_CHARS:
            return [text.strip()] if text.strip() else []

        chunks = []
        start = 0

        while start < len(text):
            end = start + MAX_CHUNK_CHARS

            # Try to break at sentence boundary
            if end < len(text):
                for sep in [". ", ".\n", "\n\n", "\n", " "]:
                    last_sep = text[start:end].rfind(sep)
                    if last_sep > MAX_CHUNK_CHARS // 2:
                        end = start + last_sep + len(sep)
                        break

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            # Move start with overlap
            start = end - CHUNK_OVERLAP
            if start >= len(text):
                break

        return chunks

    def process_batch(
        self,
        folders: List[S3ConversationFolder],
        job_id: Optional[Union[str, uuid.UUID]] = None,
    ) -> ProcessingStats:
        """Process a batch of folders."""
        self.stats = ProcessingStats()

        # Update job status if provided
        if job_id:
            self._update_job_status(
                job_id, "processing", {"total_folders": len(folders)}
            )

        for i, folder in enumerate(folders):
            try:
                self.process_folder(folder)

                # Update progress periodically
                if job_id and (i + 1) % 10 == 0:
                    self._update_job_status(
                        job_id,
                        "processing",
                        {"processed": i + 1, "total": len(folders)},
                    )

            except Exception as e:
                logger.error(f"Error processing folder {folder.name}: {e}")
                self.stats.errors += 1

        # Update job status to completed
        if job_id:
            status = "completed" if self.stats.errors == 0 else "completed_with_errors"
            self._update_job_status(job_id, status, self.stats.__dict__)

        return self.stats

    def _update_job_status(
        self, job_id: Union[str, uuid.UUID], status: str, stats: Dict[str, Any]
    ) -> None:
        """Update ingestion job status in database."""
        try:
            with self.SessionLocal() as session:
                normalized_id = self._coerce_job_id(job_id)
                job = session.get(IngestJob, normalized_id)
                if job:
                    job.status = status
                    job.stats = stats
                    job.updated_at = datetime.now(timezone.utc)
                    session.commit()
        except Exception as e:
            logger.error(f"Failed to update job status: {e}")

    def _coerce_job_id(self, job_id: Union[str, uuid.UUID]) -> uuid.UUID:
        """Normalize job identifiers to UUID objects."""
        if isinstance(job_id, uuid.UUID):
            return job_id
        return uuid.UUID(str(job_id))

    def run_full_ingestion(
        self,
        prefix: str = "raw/outlook/",
        limit: Optional[int] = None,
        job_id: Optional[Union[str, uuid.UUID]] = None,
    ) -> ProcessingStats:
        """
        Run full ingestion from S3.

        Args:
            prefix: S3 prefix to scan
            limit: Max folders to process (None = all)
            job_id: Optional job ID for tracking

        Returns:
            ProcessingStats with results
        """
        logger.info(f"Starting full ingestion from {prefix}")

        # Create job record if not provided
        job_uuid: uuid.UUID
        if job_id:
            job_uuid = self._coerce_job_id(job_id)
        else:
            job_uuid = uuid.uuid4()
            with self.SessionLocal() as session:
                job = IngestJob(
                    job_id=job_uuid,
                    tenant_id=self.tenant_id,
                    source_type="s3",
                    source_uri=prefix,
                    status="pending",
                    metadata_={"prefix": prefix},
                    options={"limit": limit} if limit else {},
                )
                session.add(job)
                session.commit()

        try:
            # List all folders
            folders = list(
                self.s3_handler.list_conversation_folders(
                    prefix=prefix, limit=limit or 10000
                )
            )

            logger.info(f"Found {len(folders)} folders to process")

            # Process in batches
            stats = self.process_batch(folders, job_uuid)

            logger.info(
                f"Ingestion complete: {stats.folders_processed} folders, "
                f"{stats.chunks_created} chunks, {stats.errors} errors"
            )

            return stats

        except Exception as e:
            logger.error(f"Ingestion failed: {e}")
            self._update_job_status(job_uuid, "failed", {"error": str(e)})
            raise


def run_ingestion_cli():
    """CLI entry point for running ingestion."""
    import argparse

    parser = argparse.ArgumentParser(description="Run S3 ingestion pipeline")
    parser.add_argument("--prefix", default="raw/outlook/", help="S3 prefix")
    parser.add_argument("--limit", type=int, help="Max folders to process")
    parser.add_argument("--tenant", default="default", help="Tenant ID")
    parser.add_argument("--batch-size", type=int, default=10, help="Batch size")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    processor = IngestionProcessor(tenant_id=args.tenant, batch_size=args.batch_size)

    stats = processor.run_full_ingestion(prefix=args.prefix, limit=args.limit)

    print("\nIngestion Results:")
    print(f"  Folders processed: {stats.folders_processed}")
    print(f"  Threads created:   {stats.threads_created}")
    print(f"  Messages created:  {stats.messages_created}")
    print(f"  Chunks created:    {stats.chunks_created}")
    print(f"  Embeddings:        {stats.embeddings_generated}")
    print(f"  Errors:            {stats.errors}")
    print(f"  Skipped:           {stats.skipped}")


if __name__ == "__main__":
    run_ingestion_cli()
