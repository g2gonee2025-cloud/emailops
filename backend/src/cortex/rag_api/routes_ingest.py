"""
Ingestion API routes.

Implements §6 of the Canonical Blueprint.
Provides endpoints for triggering and monitoring ingestion jobs.
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from cortex.context import tenant_id_ctx
from cortex.ingestion.s3_source import S3ConversationFolder, S3SourceHandler
from fastapi import APIRouter, BackgroundTasks, HTTPException, Query
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ingest", tags=["ingestion"])


# -----------------------------------------------------------------------------
# Request/Response Models
# -----------------------------------------------------------------------------


class IngestFromS3Request(BaseModel):
    """Request to ingest conversations from S3/Spaces."""

    prefix: str = Field(default="Outlook/", description="S3 prefix to scan")
    limit: Optional[int] = Field(default=None, description="Max folders to process")
    dry_run: bool = Field(
        default=False, description="If true, only list folders without ingesting"
    )


class IngestFromS3Response(BaseModel):
    """Response from S3 ingestion request."""

    job_id: str
    status: str
    folders_found: int
    folders_to_process: List[str] = []
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
    folders: List[Dict[str, Any]]
    count: int


# -----------------------------------------------------------------------------
# In-memory job tracking (for MVP - replace with Redis/DB in production)
# -----------------------------------------------------------------------------

_active_jobs: Dict[str, Dict[str, Any]] = {}


# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------


@router.get("/s3/folders", response_model=ListS3FoldersResponse)
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
        handler = S3SourceHandler()
        folders = []

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
        logger.error(f"Failed to list S3 folders: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to list S3 folders: {str(e)}"
        )


@router.post("/s3/start", response_model=IngestFromS3Response)
async def start_s3_ingestion(
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
        handler = S3SourceHandler()
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
            "started_at": datetime.now(timezone.utc).isoformat(),
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
        logger.error(f"Failed to start S3 ingestion: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to start ingestion: {str(e)}"
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


async def _process_s3_folders_with_embeddings(
    job_id: str,
    tenant_id: str,
    folders: List[S3ConversationFolder],
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
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "message": f"Processed {stats.folders_processed} folders, created {stats.chunks_created} chunks with embeddings",
            }
        )

        logger.info(
            f"Ingestion job {job_id} completed: {stats.folders_processed} folders, {stats.chunks_created} chunks"
        )

    except Exception as e:
        logger.error(f"Ingestion job {job_id} failed: {e}", exc_info=True)
        job_data.update(
            {
                "status": "failed",
                "error": str(e),
                "message": f"Ingestion failed: {str(e)}",
            }
        )
