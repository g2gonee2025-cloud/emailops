"""
Ingestion API routes.

Implements §6 of the Canonical Blueprint.
Provides endpoints for triggering and monitoring ingestion jobs.
"""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime
from typing import Any, Literal

from cortex.context import tenant_id_ctx
from cortex.ingestion.s3_source import S3ConversationFolder, S3SourceHandler
from fastapi import APIRouter, BackgroundTasks, HTTPException, Query
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ingest", tags=["ingestion"])


# -----------------------------------------------------------------------------
# Request/Response Models
# -----------------------------------------------------------------------------


class IngestFromS3Request(BaseModel):
    """Request to ingest conversations from S3/Spaces."""

    prefix: str = Field(default="Outlook/", description="S3 prefix to scan")
    limit: int | None = Field(default=None, description="Max folders to process")
    dry_run: bool = Field(
        default=False, description="If true, only list folders without ingesting"
    )


class IngestFromS3Response(BaseModel):
    """Response from S3 ingestion request."""

    job_id: str
    status: str
    folders_found: int
    folders_to_process: list[str] = Field(default_factory=list)
    message: str


class IngestStatusResponse(BaseModel):
    """Ingestion job status."""

    job_id: str
    status: str
    folders_processed: int = 0
    threads_created: int = 0
    chunks_created: int = 0
    embeddings_generated: int = 0
    errors: int = 0
    skipped: int = 0
    message: str = ""


class ListS3FoldersResponse(BaseModel):
    """Response listing S3 conversation folders."""

    prefix: str
    folders: list[dict[str, Any]]
    count: int


class PushDocument(BaseModel):
    """Document payload for push ingestion."""

    document_id: str | None = Field(
        default=None, description="External document ID for idempotent upserts"
    )
    title: str | None = Field(default=None, description="Optional document title")
    source: str | None = Field(
        default=None, description="Source identifier or storage URI"
    )
    text_type: Literal["email", "attachment"] = Field(
        default="attachment", description="Preprocessing profile for the document"
    )
    section_path: str | None = Field(
        default=None, description="Optional section path for chunk metadata"
    )
    text: str = Field(..., min_length=1, description="Document text to index")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Optional metadata to store with the document"
    )


class PushIngestRequest(BaseModel):
    """Request to push documents into the index."""

    documents: list[PushDocument] = Field(
        default_factory=list, description="Documents to index"
    )
    generate_embeddings: bool = Field(
        default=True, description="Generate embeddings for ingested chunks"
    )
    chunk_size: int | None = Field(
        default=None, ge=128, description="Override max tokens per chunk"
    )
    chunk_overlap: int | None = Field(
        default=None, ge=0, description="Override overlap tokens per chunk"
    )
    min_tokens: int | None = Field(
        default=None, ge=1, description="Override minimum tokens per chunk"
    )


class PushIngestResponse(BaseModel):
    """Response for push ingestion requests."""

    job_id: str
    documents_received: int
    documents_ingested: int
    chunks_created: int
    embeddings_generated: int
    errors: list[str] = Field(default_factory=list)
    message: str


# -----------------------------------------------------------------------------
# In-memory job tracking (for MVP - replace with Redis/DB in production)
# -----------------------------------------------------------------------------

_active_jobs: dict[str, dict[str, Any]] = {}


# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------


@router.get("/s3/folders", response_model=ListS3FoldersResponse)
def list_s3_folders(
    prefix: str = Query(default="Outlook/", description="S3 prefix to list"),
    limit: int = Query(default=100, ge=1, le=1000, description="Max folders to return"),
):
    """
    List conversation folders in S3/Spaces.

    Returns a list of folders under the specified prefix that contain
    conversation data ready for ingestion.
    """
    try:
        folders = []
        with S3SourceHandler() as handler:
            for folder in handler.list_conversation_folders(prefix=prefix, limit=limit):
                folders.append(
                    {
                        "prefix": folder.prefix,
                        "name": folder.name,
                        "file_count": len(folder.files),
                        "has_conversation": any(
                            "conversation.txt" in f for f in folder.files
                        ),
                        "has_manifest": any("manifest.json" in f for f in folder.files),
                    }
                )

        return ListS3FoldersResponse(
            prefix=prefix,
            folders=folders,
            count=len(folders),
        )
    except Exception as e:
        logger.error("Failed to list S3 folders: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to list S3 folders: {e!s}")


@router.post("/s3/start", response_model=IngestFromS3Response)
def start_s3_ingestion(
    request: IngestFromS3Request,
    background_tasks: BackgroundTasks,
):
    """
    Start ingestion of conversations from S3/Spaces.

    This endpoint:
    1. Scans the specified S3 prefix for conversation folders
    2. Creates ingestion jobs for each folder
    3. Processes jobs in the background with embedding generation

    Use dry_run=true to preview what would be ingested without actually processing.
    """
    job_id = str(uuid.uuid4())

    try:
        with S3SourceHandler() as handler:
            folders = list(
                handler.list_conversation_folders(
                    prefix=request.prefix,
                    limit=request.limit,
                )
            )

        folder_names = [f.name for f in folders]

        if request.dry_run:
            return IngestFromS3Response(
                job_id=job_id,
                status="dry_run",
                folders_found=len(folders),
                folders_to_process=folder_names[:50],  # Limit response size
                message=f"Dry run: Found {len(folders)} folders to process",
            )

        # Initialize job tracking
        _active_jobs[job_id] = {
            "status": "started",
            "started_at": datetime.now(UTC).isoformat(),
            "folders_found": len(folders),
            "folders_processed": 0,
            "threads_created": 0,
            "chunks_created": 0,
            "embeddings_generated": 0,
            "errors": 0,
            "skipped": 0,
        }

        tenant_id = tenant_id_ctx.get()

        # Process in background
        background_tasks.add_task(
            _process_s3_folders_with_embeddings,
            job_id=job_id,
            tenant_id=tenant_id,
            folders=folders,
        )

        return IngestFromS3Response(
            job_id=job_id,
            status="started",
            folders_found=len(folders),
            folders_to_process=folder_names[:50],
            message=f"Started ingestion of {len(folders)} folders with embedding generation",
        )

    except Exception as e:
        logger.error("Failed to start S3 ingestion: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to start ingestion: {e!s}")


@router.post("", response_model=PushIngestResponse)
async def push_ingest(request: PushIngestRequest) -> PushIngestResponse:
    """
    Push documents into the index.

    Allows external systems to programmatically add content to the index.
    """
    if not request.documents:
        raise HTTPException(status_code=400, detail="No documents provided")

    job_id = str(uuid.uuid4())
    tenant_id = tenant_id_ctx.get()

    return await run_in_threadpool(
        _process_push_ingest,
        request,
        tenant_id,
        job_id,
    )


@router.get("/status/{job_id}", response_model=IngestStatusResponse)
async def get_ingestion_status(job_id: str):
    """Get the status of an ingestion job."""
    if job_id not in _active_jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    job_data = _active_jobs[job_id]
    return IngestStatusResponse(
        job_id=job_id,
        status=job_data.get("status", "unknown"),
        folders_processed=job_data.get("folders_processed", 0),
        threads_created=job_data.get("threads_created", 0),
        chunks_created=job_data.get("chunks_created", 0),
        embeddings_generated=job_data.get("embeddings_generated", 0),
        errors=job_data.get("errors", 0),
        skipped=job_data.get("skipped", 0),
        message=job_data.get("message", ""),
    )


# -----------------------------------------------------------------------------
# Background Processing with Embeddings
# -----------------------------------------------------------------------------


def _process_s3_folders_with_embeddings(
    job_id: str,
    tenant_id: str,
    folders: list[S3ConversationFolder],
):
    """Process S3 folders with embedding generation."""
    from cortex.ingestion.processor import IngestionProcessor

    job_data = _active_jobs.get(job_id)
    if not job_data:
        logger.error(f"Job {job_id} not found in active jobs")
        return

    try:
        processor = IngestionProcessor(tenant_id=tenant_id)
        stats = processor.process_batch(folders, job_id)

        # Update job data with final stats
        job_data.update(
            {
                "status": "completed" if stats.errors == 0 else "completed_with_errors",
                "folders_processed": stats.folders_processed,
                "threads_created": stats.threads_created,
                "chunks_created": stats.chunks_created,
                "embeddings_generated": stats.embeddings_generated,
                "errors": stats.errors,
                "skipped": stats.skipped,
                "completed_at": datetime.now(UTC).isoformat(),
                "message": f"Processed {stats.folders_processed} folders, created {stats.chunks_created} chunks with embeddings",
            }
        )

        logger.info(
            f"Ingestion job {job_id} completed: {stats.folders_processed} folders, {stats.chunks_created} chunks"
        )

    except Exception as e:
        logger.error("Ingestion job %s failed: %s", job_id, e, exc_info=True)
        job_data.update(
            {
                "status": "failed",
                "error": str(e),
                "message": f"Ingestion failed: {e!s}",
            }
        )


def _generate_stable_id(namespace: uuid.UUID, *args: str) -> uuid.UUID:
    """
    Generate a stable UUID-5 based on a namespace and a list of string arguments.
    """
    joined = ":".join(str(a) for a in args)
    return uuid.uuid5(namespace, joined)


def _process_push_ingest(
    request: PushIngestRequest,
    tenant_id: str,
    job_id: str,
) -> PushIngestResponse:
    from cortex.chunking.chunker import ChunkingInput, chunk_text
    from cortex.config.loader import get_config
    from cortex.db.models import Chunk
    from cortex.db.session import SessionLocal, set_session_tenant
    from cortex.embeddings.client import EmbeddingsClient
    from cortex.ingestion.text_preprocessor import get_text_preprocessor
    from cortex.ingestion.writer import ChunkRecord, DBWriter, ensure_chunk_metadata
    from sqlalchemy import delete

    config = get_config()
    preprocessor = get_text_preprocessor()

    chunk_size = request.chunk_size or config.processing.chunk_size
    chunk_overlap = request.chunk_overlap or config.processing.chunk_overlap
    min_tokens = request.min_tokens or 25

    embeddings_client = EmbeddingsClient() if request.generate_embeddings else None

    documents_ingested = 0
    chunks_created = 0
    embeddings_generated = 0
    errors: list[str] = []

    tenant_ns = uuid.uuid5(uuid.NAMESPACE_DNS, f"tenant:{tenant_id}")

    # Data to be written in a single transaction
    conversations_to_write = []
    chunks_to_write = []
    cleanup_info: dict[uuid.UUID, list[uuid.UUID]] = {}

    for document in request.documents:
        try:
            doc_key = document.document_id or str(uuid.uuid4())
            conversation_id = (
                _generate_stable_id(tenant_ns, "document", doc_key)
                if document.document_id
                else uuid.uuid4()
            )

            cleaned_text, cleaning_meta = preprocessor.prepare_for_indexing(
                document.text,
                text_type=document.text_type,
                tenant_id=tenant_id,
                metadata=document.metadata,
            )

            section_path = document.section_path or f"document:{doc_key}"
            chunk_models = chunk_text(
                ChunkingInput(
                    text=cleaned_text,
                    section_path=section_path,
                    max_tokens=chunk_size,
                    min_tokens=min_tokens,
                    overlap_tokens=chunk_overlap,
                )
            )

            chunk_texts = [chunk.text for chunk in chunk_models]
            embeddings: list[list[float]] = []
            if embeddings_client and chunk_texts:
                embeddings = embeddings_client.embed_batch(chunk_texts)

            current_chunk_ids = []
            for idx, chunk in enumerate(chunk_models):
                content_hash = chunk.metadata.get("content_hash", "")
                chunk_id = _generate_stable_id(
                    tenant_ns,
                    "chunk",
                    str(conversation_id),
                    str(chunk.position),
                    content_hash,
                )
                current_chunk_ids.append(chunk_id)

                extra_data = {
                    **chunk.metadata,
                    "document_id": doc_key,
                    "document_metadata": document.metadata,
                }

                chunk_data = {
                    "chunk_id": chunk_id,
                    "conversation_id": conversation_id,
                    "text": chunk.text,
                    "chunk_type": chunk.chunk_type,
                    "position": chunk.position,
                    "char_start": chunk.char_start,
                    "char_end": chunk.char_end,
                    "section_path": chunk.section_path,
                    "extra_data": extra_data,
                    "source": "external",
                }
                if embeddings:
                    embedding_list = embeddings[idx]
                    chunk_data["embedding"] = (
                        embedding_list.tolist()
                        if hasattr(embedding_list, "tolist")
                        else list(embedding_list)
                    )

                chunks_to_write.append(chunk_data)

            conversation_extra = {
                "document_id": doc_key,
                "source": document.source,
                "metadata": document.metadata,
                "ingest_type": "push",
                "text_type": document.text_type,
                **cleaning_meta,
            }

            conversation_data = {
                "conversation_id": conversation_id,
                "tenant_id": tenant_id,
                "folder_name": document.document_id
                or document.title
                or str(conversation_id),
                "subject": document.title,
                "storage_uri": document.source,
                "extra_data": conversation_extra,
            }
            conversations_to_write.append(conversation_data)
            cleanup_info[conversation_id] = current_chunk_ids

            documents_ingested += 1
            chunks_created += len(chunk_models)
            if embeddings:
                embeddings_generated += len(embeddings)

        except Exception as exc:
            error_ref = document.document_id or document.title or "document"
            errors.append(f"{error_ref}: {exc}")
            logger.error(
                "Push ingestion failed for %s: %s", error_ref, exc, exc_info=True
            )

    if documents_ingested > 0:
        try:
            with SessionLocal() as session:
                set_session_tenant(session, tenant_id)
                writer = DBWriter(session)

                for conv_data in conversations_to_write:
                    writer.write_conversation(**conv_data)

                for chunk_data in chunks_to_write:
                    source = chunk_data.get("source", "attachment")
                    chunk_data = ensure_chunk_metadata(chunk_data, source=source)
                    record = ChunkRecord(
                        tenant_id=tenant_id,
                        chunk_id=chunk_data["chunk_id"],
                        conversation_id=chunk_data["conversation_id"],
                        text=chunk_data["text"],
                        chunk_type=chunk_data.get("chunk_type", "message_body"),
                        embedding=chunk_data.get("embedding"),
                        position=chunk_data.get("position", 0),
                        char_start=chunk_data.get("char_start", 0),
                        char_end=chunk_data.get("char_end", 0),
                        section_path=chunk_data.get("section_path"),
                        extra_data=chunk_data.get("extra_data"),
                    )
                    writer.write_chunk(record)

                for cid, cids in cleanup_info.items():
                    stmt = (
                        delete(Chunk).where(Chunk.conversation_id == cid)
                        if not cids
                        else delete(Chunk).where(
                            Chunk.conversation_id == cid, Chunk.chunk_id.notin_(cids)
                        )
                    )
                    session.execute(stmt)
                session.commit()
        except Exception as exc:
            errors.append(f"Database transaction failed: {exc}")
            logger.error("Push ingestion DB transaction failed: %s", exc, exc_info=True)
            documents_ingested = chunks_created = embeddings_generated = 0

    message = f"Ingested {documents_ingested}/{len(request.documents)} documents"
    return PushIngestResponse(
        job_id=job_id,
        documents_received=len(request.documents),
        documents_ingested=documents_ingested,
        chunks_created=chunks_created,
        embeddings_generated=embeddings_generated,
        errors=errors,
        message=message,
    )
