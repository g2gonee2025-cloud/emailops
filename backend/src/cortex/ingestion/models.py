"""
Pydantic models for ingestion jobs and summaries.
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class Problem(BaseModel):
    folder: str
    issue: str


class IngestJobRequest(BaseModel):
    job_id: uuid.UUID
    tenant_id: str
    source_type: Literal["s3", "sftp", "local_upload"]
    source_uri: str
    options: Dict[str, Any] = Field(default_factory=dict)


class IngestJobSummary(BaseModel):
    job_id: uuid.UUID
    tenant_id: str
    messages_total: int = 0
    messages_ingested: int = 0
    messages_failed: int = 0
    attachments_total: int = 0
    attachments_parsed: int = 0
    attachments_failed: int = 0

    # Batch processing stats
    folders_processed: int = 0
    threads_created: int = 0
    chunks_created: int = 0
    embeddings_generated: int = 0
    errors: int = 0
    skipped: int = 0

    problems: List[Problem] = Field(default_factory=list)
    aborted_reason: Optional[str] = None
