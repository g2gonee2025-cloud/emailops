"""
Ingestion API routes.

Implements §6 of the Canonical Blueprint.
Provides endpoints for triggering and monitoring ingestion jobs.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Literal

import redis.asyncio as redis
from cortex.common.redis import get_redis
from cortex.config.loader import get_config
from cortex.context import tenant_id_ctx
from cortex.ingestion.s3_source import S3ConversationFolder, S3SourceHandler
from cortex.security.auth import get_current_user
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
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
    limit: int | None = Field(default=None, ge=1, description="Max folders to process")
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
# Endpoints
# -----------------------------------------------------------------------------


@router.get(
    "/s3/folders",
    response_model=ListS3FoldersResponse,
    dependencies=[Depends(get_current_user)],
)
async def list_s3_folders(
    prefix: str = Query(default="Outlook/", description="S3 prefix to list"),
    limit: int = Query(default=100, ge=1, le=1000, description="Max folders to return"),
):
    """
    List conversation folders in S3/Spaces.

    Returns a list of folders under the specified prefix that contain
    conversation data ready for ingestion.
    """
    try:

        def sync_list_folders():
            with S3SourceHandler() as handler:
                return list(
                    handler.list_conversation_folders(prefix=prefix, limit=limit)
                )

        s3_folders = await run_in_threadpool(sync_list_folders)

        def _root_files(folder: S3ConversationFolder) -> set[str]:
            root_files: set[str] = set()
            prefix_key = folder.prefix
            for key in folder.files:
                if not key.startswith(prefix_key):
                    continue
                relative = key[len(prefix_key) :]
                if "/" in relative:
                    continue
                root_files.add(relative.lower())
            return root_files

        response_folders = []
        for folder in s3_folders:
            root_files = _root_files(folder)
            response_folders.append(
                {
                    "prefix": folder.prefix,
                    "name": folder.name,
                    "file_count": len(folder.files),
                    "has_conversation": "conversation.txt" in root_files,
                    "has_manifest": "manifest.json" in root_files,
                }
            )

        return ListS3FoldersResponse(
            prefix=prefix,
            folders=response_folders,
            count=len(response_folders),
        )
    except Exception as e:
        logger.error("Failed to list S3 folders: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to list S3 folders: {e!s}")


@router.post(
    "/s3/start",
    response_model=IngestFromS3Response,
    dependencies=[Depends(get_current_user)],
)
async def start_s3_ingestion(
    request: IngestFromS3Request,
    background_tasks: BackgroundTasks,
    redis_client: redis.Redis = Depends(get_redis),
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

        def sync_list_folders():
            with S3SourceHandler() as handler:
                return list(
                    handler.list_conversation_folders(
                        prefix=request.prefix, limit=request.limit
                    )
                )

        folders = await run_in_threadpool(sync_list_folders)
        folder_names = [f.name for f in folders]

        if request.dry_run:
            return IngestFromS3Response(
                job_id=job_id,
                status="dry_run",
                folders_found=len(folders),
                folders_to_process=folder_names[:50],  # Limit response size
                message=f"Dry run: Found {len(folders)} folders to process",
            )

        tenant_id = _require_tenant_id()
        job_key = f"tenant:{tenant_id}:ingest_job:{job_id}"

        job_data = {
            "job_id": job_id,
            "status": "started",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "folders_found": len(folders),
            "folders_processed": 0,
            "threads_created": 0,
            "chunks_created": 0,
            "embeddings_generated": 0,
            "errors": 0,
            "skipped": 0,
        }
        await redis_client.set(job_key, json.dumps(job_data), ex=86400)

        # Process in background
        redis_url = str(get_config().redis.url)
        background_tasks.add_task(
            _run_s3_ingest_background,
            job_id=job_id,
            tenant_id=tenant_id,
            folders=folders,
            redis_url=redis_url,
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


@router.post(
    "",
    response_model=PushIngestResponse,
    dependencies=[Depends(get_current_user)],
)
async def push_ingest(request: PushIngestRequest) -> PushIngestResponse:
    """
    Push documents into the index.

    Allows external systems to programmatically add content to the index.
    """
    if not request.documents:
        raise HTTPException(status_code=400, detail="No documents provided")

    job_id = str(uuid.uuid4())
    tenant_id = _require_tenant_id()

    return await run_in_threadpool(
        _process_push_ingest,
        request,
        tenant_id,
        job_id,
    )


@router.get(
    "/status/{job_id}",
    response_model=IngestStatusResponse,
    dependencies=[Depends(get_current_user)],
)
async def get_ingestion_status(
    job_id: str,
    redis_client: redis.Redis = Depends(get_redis),
):
    """Get the status of an ingestion job."""
    tenant_id = _require_tenant_id()
    job_key = f"tenant:{tenant_id}:ingest_job:{job_id}"

    job_data_raw = await redis_client.get(job_key)
    if not job_data_raw:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    try:
        job_data = json.loads(job_data_raw)
    except json.JSONDecodeError:
        logger.error("Failed to decode JSON for job %s", job_id, exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Could not parse status for job {job_id}"
        )
    # Validate that job_data is a dict
    if not isinstance(job_data, dict):
        logger.error("Invalid job data format for job %s: expected dict", job_id)
        raise HTTPException(
            status_code=500, detail=f"Invalid job data format for job {job_id}"
        )

    # Explicitly map fields to the response model to avoid validation errors
    # from extra fields like 'started_at' or 'folders_found' in the Redis cache.
    return IngestStatusResponse(
        job_id=job_data.get("job_id", job_id),
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


def _run_s3_ingest_background(
    job_id: str,
    tenant_id: str,
    folders: list[S3ConversationFolder],
    redis_url: str,
) -> None:
    asyncio.run(
        _process_s3_folders_with_embeddings(
            job_id=job_id,
            tenant_id=tenant_id,
            folders=folders,
            redis_url=redis_url,
        )
    )


async def _process_s3_folders_with_embeddings(
    job_id: str,
    tenant_id: str,
    folders: list[S3ConversationFolder],
    redis_url: str,
):
    """Process S3 folders with embedding generation."""
    from cortex.ingestion.processor import IngestionProcessor

    redis_client: redis.Redis | None = None
    job_key = f"tenant:{tenant_id}:ingest_job:{job_id}"
    job_data: dict[str, Any] = {}
    token = None
    try:
        redis_client = redis.from_url(redis_url)
        job_data_raw = await redis_client.get(job_key)
        if not job_data_raw:
            error_job_data = {
                "job_id": job_id,
                "status": "failed",
                "error": "Job not found in Redis",
                "message": "Critical error: job not found.",
                "completed_at": datetime.now(timezone.utc).isoformat(),
            }
            await redis_client.set(job_key, json.dumps(error_job_data), ex=86400)
            logger.error("Job %s not found in Redis for tenant %s", job_id, tenant_id)
            return
        job_data = json.loads(job_data_raw)
    except Exception as e:
        logger.error(
            "Failed to load or parse job %s from Redis: %s", job_id, e, exc_info=True
        )
        if redis_client is not None:
            error_job_data = {
                "status": "failed",
                "error": f"Failed to load job from Redis: {e}",
                "message": "Critical error: could not read job data.",
                "completed_at": datetime.now(timezone.utc).isoformat(),
            }
            await redis_client.set(job_key, json.dumps(error_job_data), ex=86400)
        return

    if not isinstance(job_data, dict):
        logger.error("Invalid job data format for job %s: expected dict", job_id)
        error_job_data = {
            "job_id": job_id,
            "status": "failed",
            "error": "Invalid job data format",
            "message": "Critical error: invalid job data format.",
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        await redis_client.set(job_key, json.dumps(error_job_data), ex=86400)
        return

    # Propagate tenant context for downstream code that may rely on it
    token = tenant_id_ctx.set(tenant_id)

    # Update status to 'processing' to give users accurate feedback that work has begun.
    job_data["status"] = "processing"
    job_data["message"] = (
        f"Processing {job_data.get('folders_found', 'unknown')} folders..."
    )
    await redis_client.set(job_key, json.dumps(job_data), keepttl=True)

    try:
        # This function should be async or run in a threadpool if it's blocking
        processor = await run_in_threadpool(IngestionProcessor, tenant_id=tenant_id)
        stats = await run_in_threadpool(processor.process_batch, folders, job_id)

        errors = _get_stat_value(stats, "errors")
        folders_processed = _get_stat_value(stats, "folders_processed")
        threads_created = _get_stat_value(stats, "threads_created")
        chunks_created = _get_stat_value(stats, "chunks_created")
        embeddings_generated = _get_stat_value(stats, "embeddings_generated")
        skipped = _get_stat_value(stats, "skipped")

        # Update job data with final stats
        job_data.update(
            {
                "status": "completed" if errors == 0 else "completed_with_errors",
                "folders_processed": folders_processed,
                "threads_created": threads_created,
                "chunks_created": chunks_created,
                "embeddings_generated": embeddings_generated,
                "errors": errors,
                "skipped": skipped,
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "message": f"Processed {folders_processed} folders, created {chunks_created} chunks with embeddings",
            }
        )
        await redis_client.set(job_key, json.dumps(job_data), ex=86400)

        logger.info(
            "Ingestion job %s completed: %d folders, %d chunks",
            job_id,
            folders_processed,
            chunks_created,
        )

    except Exception as e:
        logger.error("Ingestion job %s failed: %s", job_id, e, exc_info=True)
        job_data.update(
            {
                "status": "failed",
                "error": str(e),
                "message": f"Ingestion failed: {e!s}",
                "completed_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        await redis_client.set(job_key, json.dumps(job_data), ex=86400)
    finally:
        if token is not None:
            tenant_id_ctx.reset(token)
        if redis_client is not None:
            await redis_client.close()
            await redis_client.connection_pool.disconnect()


def _get_stat_value(stats: Any, key: str, default: int = 0) -> int:
    if isinstance(stats, dict):
        value = stats.get(key, default)
    else:
        value = getattr(stats, key, default)
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _require_tenant_id() -> str:
    tenant_id = tenant_id_ctx.get()
    if not isinstance(tenant_id, str) or not tenant_id.strip():
        raise HTTPException(status_code=401, detail="Tenant context missing")
    return tenant_id


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

    chunk_size = (
        request.chunk_size
        if request.chunk_size is not None
        else config.processing.chunk_size
    )
    chunk_overlap = (
        request.chunk_overlap
        if request.chunk_overlap is not None
        else config.processing.chunk_overlap
    )
    min_tokens = request.min_tokens if request.min_tokens is not None else 25

    if chunk_overlap >= chunk_size:
        raise HTTPException(
            status_code=400, detail="chunk_overlap must be less than chunk_size"
        )
    if min_tokens > chunk_size:
        raise HTTPException(
            status_code=400,
            detail="min_tokens must be less than or equal to chunk_size",
        )

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
            chunk_source = (
                "email_body" if document.text_type == "email" else "attachment"
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
                embeddings = embeddings_client.embed_texts(chunk_texts)
            if embeddings and len(embeddings) != len(chunk_models):
                raise ValueError("Embedding count mismatch")

            doc_chunks: list[dict[str, Any]] = []
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
                    "source": chunk_source,
                }
                if embeddings:
                    embedding_list = embeddings[idx]
                    if embedding_list:
                        chunk_data["embedding"] = (
                            embedding_list.tolist()
                            if hasattr(embedding_list, "tolist")
                            else list(embedding_list)
                        )

                doc_chunks.append(chunk_data)

            conversation_extra = {
                "document_id": doc_key,
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
            chunks_to_write.extend(doc_chunks)
            cleanup_info[conversation_id] = current_chunk_ids

            documents_ingested += 1
            chunks_created += len(chunk_models)
            if embeddings:
                embeddings_generated += sum(1 for embedding in embeddings if embedding)

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
                    processed_chunk = ensure_chunk_metadata(chunk_data, source=source)
                    record = ChunkRecord(
                        tenant_id=tenant_id,
                        chunk_id=processed_chunk["chunk_id"],
                        conversation_id=processed_chunk["conversation_id"],
                        text=processed_chunk["text"],
                        chunk_type=processed_chunk.get("chunk_type", "message_body"),
                        embedding=processed_chunk.get("embedding"),
                        position=processed_chunk.get("position", 0),
                        char_start=processed_chunk.get("char_start", 0),
                        char_end=processed_chunk.get("char_end", 0),
                        section_path=processed_chunk.get("section_path"),
                        extra_data=processed_chunk.get("extra_data"),
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
