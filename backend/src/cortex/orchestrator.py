"""
Unified Pipeline Orchestrator.

Implements the "Atomic Batch" processing logic defined in the Unified Pipeline Design.
Orchestrates: Source -> Ingest -> Embed
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from cortex.indexer import Indexer
from cortex.ingestion.processor import IngestionProcessor
from cortex.ingestion.s3_source import S3SourceHandler

logger = logging.getLogger(__name__)


@dataclass
class PipelineStats:
    folders_found: int = 0
    folders_processed: int = 0
    folders_skipped: int = 0
    folders_failed: int = 0
    chunks_created: int = 0
    embeddings_generated: int = 0
    start_time: float = field(default_factory=time.time)
    errors_list: list = field(default_factory=list)  # [{folder, reason}]

    @property
    def duration_seconds(self) -> float:
        return time.time() - self.start_time


class PipelineOrchestrator:
    """
    Orchestrates the end-to-end ingestion pipeline.

    Acts as the high-level controller that coordinates:
    1. Discovery (S3SourceHandler)
    2. Ingestion (IngestionProcessor)
    3. Embedding (EmbeddingsClient / Indexer)
    """

    def __init__(
        self,
        tenant_id: str = "default",
        auto_embed: bool = False,
        concurrency: int = 4,
        dry_run: bool = False,
    ):
        self.tenant_id = tenant_id
        self.auto_embed = auto_embed
        self.concurrency = concurrency
        self.dry_run = dry_run
        self.processor = IngestionProcessor(tenant_id=tenant_id)
        if not self.dry_run:
            self.s3_handler = S3SourceHandler()
        else:
            self.s3_handler = None
        self.indexer = Indexer(concurrency=concurrency)
        self.stats = PipelineStats()

    def run(
        self, source_prefix: str = "Outlook/", limit: int | None = None
    ) -> PipelineStats:
        """
        Run the full pipeline for a given S3 prefix.
        """
        logger.info(
            f"Pipeline started for tenant={self.tenant_id} prefix={source_prefix}"
        )

        # 1. Discovery (Lazy iterator)
        if self.dry_run:
            folders_iter = []
        else:
            folders_iter = self.s3_handler.list_conversation_folders(
                prefix=source_prefix, limit=limit
            )

        # 2. Enqueueing
        logger.info("Pipeline: Discovering and enqueuing jobs...")
        enqueued_count = 0
        for folder in folders_iter:
            try:
                self._enqueue_ingest_job(folder)
                enqueued_count += 1
            except Exception as e:
                logger.error(f"Failed to enqueue job for folder {folder}: {e}")
                self.stats.folders_failed += 1

        self.stats.folders_found = enqueued_count
        self.stats.folders_processed = enqueued_count  # For CLI output clarity

        total_time = self.stats.duration_seconds
        logger.info(f"Enqueued {enqueued_count} jobs in {total_time:.2f}s.")
        logger.info(
            f"Pipeline complete in {total_time:.2f}s. "
            f"Processed: {self.stats.folders_processed}, "
            f"Skipped: {self.stats.folders_skipped}, "
            f"Failed: {self.stats.folders_failed}"
        )
        return self.stats

    def _enqueue_ingest_job(self, folder: str) -> None:
        """
        Enqueues an ingestion job for a single folder.
        """
        from cortex.ingestion.models import IngestJobRequest
        from cortex.queue import get_queue

        folder_name = (
            getattr(folder, "name", None)
            or getattr(folder, "prefix", None)
            or str(folder)
        )

        # Dry-run mode: log and skip
        if self.dry_run:
            logger.info(f"[DRY-RUN] Would enqueue job for: {folder_name}")
            return

        try:
            # Create the job request payload
            job_request = IngestJobRequest(
                tenant_id=self.tenant_id,
                source_uri=folder_name,
                source_type="s3",  # Assuming S3 source for now
                auto_embed=self.auto_embed,
            )

            # Get queue and enqueue
            queue = get_queue()
            job_id = queue.enqueue(
                job_type="ingest",
                payload=job_request.dict(),
                priority=5,  # Normal priority
            )
            logger.info(f"Enqueued ingest job {job_id} for folder {folder_name}")

        except Exception as e:
            self.stats.folders_failed += 1
            self.stats.errors_list.append({"folder": folder_name, "reason": str(e)})
            logger.exception(f"Failed to enqueue ingest job for {folder_name}: {e}")
