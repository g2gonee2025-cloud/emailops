"""
CLI wrapper/orchestrator around mailroom.process_job.

Simplified for new Conversation-based schema (no IngestJob table).
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from typing import Any, List, Optional

from cortex.config.loader import get_config
from cortex.ingestion.mailroom import process_job
from cortex.ingestion.models import IngestJobRequest, IngestJobSummary
from cortex.ingestion.s3_source import S3SourceHandler

logger = logging.getLogger(__name__)


@dataclass
class ProcessingStats:
    folders_processed: int = 0
    jobs_created: int = 0
    errors: int = 0


def _derive_status(summary: IngestJobSummary) -> str:
    if summary.aborted_reason:
        return "failed"
    if summary.messages_failed or summary.attachments_failed:
        return "completed_with_errors"
    return "completed"


class IngestionProcessor:
    """Thin orchestrator that hands off ingestion to mailroom.process_job."""

    def __init__(self, tenant_id: str = "default") -> None:
        self.config = get_config()
        self.tenant_id = tenant_id
        self.s3_handler = S3SourceHandler()
        self.stats = ProcessingStats()

    def _build_job_request(self, folder_prefix: str) -> IngestJobRequest:
        return IngestJobRequest(
            job_id=uuid.uuid4(),
            tenant_id=self.tenant_id,
            source_type="s3",
            source_uri=folder_prefix,
            options={"prefix": folder_prefix},
        )

    def process_folder(self, folder_prefix: str) -> Optional[IngestJobSummary]:
        """Process a single folder and return summary."""
        job_request = self._build_job_request(folder_prefix)
        try:
            summary = process_job(job_request)
            self.stats.folders_processed += 1
            self.stats.jobs_created += 1
            return summary
        except Exception as exc:
            logger.error("Ingestion failed for %s: %s", folder_prefix, exc)
            self.stats.errors += 1
            return None

    def run_full_ingestion(
        self, prefix: str = "Outlook/", limit: Optional[int] = None
    ) -> List[IngestJobSummary]:
        """Run full ingestion with parallel workers for GPU saturation."""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        num_workers = self.config.processing.num_workers
        logger.info(
            "Starting ingestion scan for prefix %s with %d workers", prefix, num_workers
        )
        summaries: List[IngestJobSummary] = []

        folders = list(
            self.s3_handler.list_conversation_folders(prefix=prefix, limit=limit)
        )
        logger.info("Found %d folders to process", len(folders))

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(self.process_folder, folder.prefix): folder
                for folder in folders
            }

            for i, future in enumerate(as_completed(futures), 1):
                try:
                    summary = future.result()
                    if summary:
                        summaries.append(summary)

                    if i % 100 == 0:
                        logger.info(
                            "Progress: %d/%d folders processed", i, len(folders)
                        )
                except Exception as exc:
                    folder = futures[future]
                    logger.error("Failed to process %s: %s", folder.prefix, exc)

        return summaries

    def process_batch(self, folders: List[Any], job_id: str) -> IngestJobSummary:
        """Process a batch of folders with parallel workers for GPU saturation."""
        import threading
        from concurrent.futures import ThreadPoolExecutor, as_completed

        num_workers = self.config.processing.num_workers
        logger.info(
            f"Processing batch of {len(folders)} folders for job {job_id} "
            f"with {num_workers} parallel workers"
        )

        agg_stats = IngestJobSummary(job_id=job_id, tenant_id=self.tenant_id)
        agg_stats.folders_processed = 0
        agg_stats.threads_created = 0
        agg_stats.chunks_created = 0
        agg_stats.embeddings_generated = 0
        agg_stats.errors = 0
        agg_stats.skipped = 0

        # Thread-safe counter for progress tracking
        stats_lock = threading.Lock()
        progress = {"count": 0}

        def process_single(folder) -> Optional[IngestJobSummary]:
            prefix = folder.prefix
            return self.process_folder(prefix)

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(process_single, f): f for f in folders}

            for future in as_completed(futures):
                folder = futures[future]
                try:
                    summary = future.result()

                    with stats_lock:
                        progress["count"] += 1
                        current_count = progress["count"]

                        if summary:
                            if summary.aborted_reason or summary.problems:
                                agg_stats.errors += 1
                            else:
                                agg_stats.folders_processed += 1

                            agg_stats.threads_created += 1
                            agg_stats.chunks_created += summary.chunks_created
                            agg_stats.embeddings_generated += (
                                summary.embeddings_generated
                            )
                        else:
                            agg_stats.errors += 1

                        # Log progress every 100 folders
                        if current_count % 100 == 0:
                            logger.info(
                                f"Progress: {current_count}/{len(folders)} folders, "
                                f"{agg_stats.chunks_created} chunks"
                            )

                except Exception as exc:
                    logger.error(f"Future failed for {folder.prefix}: {exc}")
                    with stats_lock:
                        agg_stats.errors += 1

        logger.info(
            f"Batch complete: {agg_stats.folders_processed} folders, "
            f"{agg_stats.chunks_created} chunks, {agg_stats.errors} errors"
        )
        return agg_stats


def run_ingestion_cli() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run S3 ingestion pipeline")
    parser.add_argument("--prefix", default="Outlook/", help="S3 prefix")
    parser.add_argument("--limit", type=int, help="Max folders to process")
    parser.add_argument("--tenant", default="default", help="Tenant ID")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    processor = IngestionProcessor(tenant_id=args.tenant)
    summaries = processor.run_full_ingestion(prefix=args.prefix, limit=args.limit)

    print("\nIngestion Results:")
    print(f"  Jobs attempted:   {len(summaries)}")
    for summary in summaries:
        status = _derive_status(summary)
        print(
            f"  {summary.job_id} | status={status} messages={summary.messages_ingested}/{summary.messages_total}"
        )


if __name__ == "__main__":
    run_ingestion_cli()
