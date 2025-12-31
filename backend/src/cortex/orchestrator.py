"""
Unified Pipeline Orchestrator.

Implements the "Atomic Batch" enqueueing logic defined in the Unified Pipeline Design.
Orchestrates: Source discovery -> Ingest job enqueue
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Literal

from cortex.ingestion.models import IngestJobRequest
from cortex.ingestion.s3_source import S3ConversationFolder, S3SourceHandler
from cortex.queue import get_queue

logger = logging.getLogger(__name__)

DEFAULT_QUEUE_PRIORITY = 5


@dataclass
class PipelineStats:
    folders_found: int = 0
    folders_enqueued: int = 0
    folders_processed: int = 0
    folders_skipped: int = 0
    folders_failed: int = 0
    chunks_created: int = 0
    embeddings_generated: int = 0
    start_time: float = field(default_factory=time.time)
    errors_list: list[dict[str, str]] = field(
        default_factory=list
    )  # [{folder, reason}]

    @property
    def duration_seconds(self) -> float:
        return time.time() - self.start_time


class PipelineOrchestrator:
    """
    Orchestrates discovery and enqueueing of ingestion jobs.

    Acts as the high-level controller that coordinates:
    1. Discovery (S3SourceHandler)
    2. Enqueueing of ingestion jobs
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
        if not self.dry_run:
            self.s3_handler = S3SourceHandler()
        else:
            self.s3_handler = None
        self.stats = PipelineStats()

    def run(
        self, source_prefix: str = "Outlook/", limit: int | None = None
    ) -> PipelineStats:
        """
        Run the full pipeline for a given S3 prefix.
        """
        self.stats = PipelineStats()
        logger.info(
            f"Pipeline started for tenant={self.tenant_id} prefix={source_prefix}"
        )

        # 1. Discovery (Lazy iterator)
        if self.dry_run or self.s3_handler is None:
            folders_iter = []
        else:
            try:
                folders_iter = self.s3_handler.list_conversation_folders(
                    prefix=source_prefix, limit=limit
                )
            except Exception as exc:
                logger.exception(
                    "Pipeline discovery failed for prefix %s", source_prefix
                )
                self.stats.folders_failed += 1
                self.stats.errors_list.append(
                    {"folder": source_prefix, "reason": str(exc)}
                )
                return self.stats

        # 2. Enqueueing
        logger.info("Pipeline: Discovering and enqueuing jobs...")
        enqueued_count = 0
        folders_found = 0
        try:
            for folder in folders_iter:
                folders_found += 1
                status, folder_name, error = self._enqueue_ingest_job(folder)
                if status == "enqueued":
                    enqueued_count += 1
                elif status == "skipped":
                    self.stats.folders_skipped += 1
                else:
                    self.stats.folders_failed += 1
                    if error:
                        self.stats.errors_list.append(
                            {"folder": folder_name, "reason": error}
                        )
        except Exception as exc:
            logger.exception("Pipeline discovery failed during iteration")
            self.stats.folders_failed += 1
            self.stats.errors_list.append({"folder": source_prefix, "reason": str(exc)})

        self.stats.folders_found = folders_found
        self.stats.folders_enqueued = enqueued_count
        self.stats.folders_processed = 0

        total_time = self.stats.duration_seconds
        logger.info(f"Enqueued {enqueued_count} jobs in {total_time:.2f}s.")
        logger.info(
            f"Pipeline complete in {total_time:.2f}s. "
            f"Enqueued: {self.stats.folders_enqueued}, "
            f"Skipped: {self.stats.folders_skipped}, "
            f"Failed: {self.stats.folders_failed}"
        )
        return self.stats

    def _enqueue_ingest_job(
        self, folder: S3ConversationFolder | str
    ) -> tuple[Literal["enqueued", "skipped", "failed"], str, str | None]:
        """
        Enqueues an ingestion job for a single folder.
        """
        folder_name = (
            getattr(folder, "name", None)
            or getattr(folder, "prefix", None)
            or str(folder)
        )

        # Dry-run mode: log and skip
        if self.dry_run:
            logger.info(f"[DRY-RUN] Would enqueue job for: {folder_name}")
            return "skipped", folder_name, None

        try:
            # Create the job request payload
            job_request = IngestJobRequest(
                job_id=uuid.uuid4(),
                tenant_id=self.tenant_id,
                source_uri=folder_name,
                source_type="s3",  # Assuming S3 source for now
                options={"auto_embed": self.auto_embed},
            )

            # Get queue and enqueue
            queue = get_queue()
            job_id = queue.enqueue(
                job_type="ingest",
                payload=job_request.model_dump(mode="json"),
                priority=DEFAULT_QUEUE_PRIORITY,
            )
            logger.info(f"Enqueued ingest job {job_id} for folder {folder_name}")
            return "enqueued", folder_name, None

        except Exception as exc:
            logger.exception("Failed to enqueue ingest job for %s", folder_name)
            return "failed", folder_name, str(exc)
