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
from cortex.ingestion.s3_source import S3ConversationFolder, S3SourceHandler

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

    def _get_existing_timestamps(self, folder_names: list[str]) -> dict[str, Any]:
        """Fetch all existing folder timestamps in a single query to prevent N+1."""
        from cortex.db.models import Conversation
        from cortex.db.session import SessionLocal
        from sqlalchemy.exc import SQLAlchemyError

        timestamps = {}
        try:
            with SessionLocal() as session:
                results = (
                    session.query(
                        Conversation.folder_name,
                        Conversation.extra_data["source_last_modified"].astext,
                    )
                    .filter(
                        Conversation.tenant_id == self.tenant_id,
                        Conversation.folder_name.in_(folder_names),
                    )
                    .all()
                )
                timestamps = {name: ts for name, ts in results if ts}
        except SQLAlchemyError as e:
            logger.error("Failed to query existing timestamps: %s", e)
            # On DB error, returning empty dict forces re-processing, which is safer
            # than skipping folders that might need updates.
        return timestamps

    def process_folder(
        self, folder: S3ConversationFolder, existing_timestamps: dict[str, Any]
    ) -> IngestJobSummary | None:
        """
        Process a single folder and return summary.
        Checks if folder needs processing based on pre-fetched last_modified timestamp.
        """
        prefix = folder.prefix
        last_modified = folder.last_modified
        folder_name = folder.name

        # Check if we can skip this folder using pre-fetched data
        if last_modified and folder_name in existing_timestamps:
            saved_mod_str = existing_timestamps[folder_name]
            from dateutil import parser

            try:
                saved_mod_dt = parser.parse(saved_mod_str)

                # Ensure timezone awareness for correct comparison
                if saved_mod_dt.tzinfo is None:
                    from datetime import timezone

                    saved_mod_dt = saved_mod_dt.replace(tzinfo=timezone.utc)

                if last_modified.tzinfo is None:
                    from datetime import timezone

                    last_modified = last_modified.replace(tzinfo=timezone.utc)

                if saved_mod_dt >= last_modified:
                    logger.info(f"Skipping unchanged folder: {prefix}")
                    return None
            except (ValueError, TypeError) as e:
                logger.warning(
                    "Could not parse saved timestamp '%s' for %s: %s",
                    saved_mod_str,
                    prefix,
                    e,
                )

        job_request = self._build_job_request(prefix)

        # Pass timestamp to job options so mailroom can persist it
        if last_modified:
            job_request.options["source_last_modified"] = last_modified.isoformat()

        try:
            summary = process_job(job_request)
            self.stats.folders_processed += 1
            self.stats.jobs_created += 1
            return summary
        except Exception as exc:  # Broad exception to catch failures in underlying job
            logger.error("Ingestion failed for %s: %s", prefix, exc, exc_info=True)
            self.stats.errors += 1
            return None

    def run_full_ingestion(
        self, prefix: str = "Outlook/", limit: int | None = None
    ) -> list[IngestJobSummary]:
        """Run full ingestion with parallel workers, avoiding N+1 DB queries."""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        num_workers = self.config.processing.num_workers
        logger.info(
            "Starting ingestion scan for prefix %s with %d workers", prefix, num_workers
        )
        summaries: list[IngestJobSummary] = []

        folders = self.s3_handler.list_conversation_folders(prefix=prefix, limit=limit)
        if not folders:
            logger.info("No folders found to process.")
            return summaries

        # Bulk fetch existing timestamps to avoid N+1 queries
        folder_names = [f.name for f in folders]
        existing_timestamps = self._get_existing_timestamps(folder_names)
        logger.info(f"Found {len(existing_timestamps)} existing folder timestamps.")

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(
                    self.process_folder, folder, existing_timestamps
                ): folder
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

    def process_batch(
        self, folders: list[S3ConversationFolder], job_id: str
    ) -> IngestJobSummary:
        """Process a batch of folders with parallel workers, avoiding N+1 DB queries."""
        import threading
        from concurrent.futures import ThreadPoolExecutor, as_completed

        num_workers = self.config.processing.num_workers
        logger.info(
            f"Processing batch of {len(folders)} folders for job {job_id} "
            f"with {num_workers} parallel workers"
        )

        agg_stats = IngestJobSummary(job_id=uuid.UUID(job_id), tenant_id=self.tenant_id)
        # ... (agg_stats initialization) ...

        # Bulk fetch existing timestamps
        folder_names = [f.name for f in folders]
        existing_timestamps = self._get_existing_timestamps(folder_names)
        logger.info(
            f"Batch pre-fetch: Found {len(existing_timestamps)} existing timestamps."
        )

        stats_lock = threading.Lock()
        progress = {"count": 0}

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(self.process_folder, f, existing_timestamps): f
                for f in folders
            }

            for future in as_completed(futures):
                folder = futures[future]
                try:
                    summary = future.result()

                    with stats_lock:
                        progress["count"] += 1
                        current_count = progress["count"]

                        if summary:
                            if summary.aborted_reason:
                                agg_stats.errors += 1
                            else:
                                agg_stats.folders_processed += 1

                            agg_stats.threads_created += 1
                            agg_stats.chunks_created += summary.chunks_created
                            agg_stats.embeddings_generated += (
                                summary.embeddings_generated
                            )
                        else:
                            agg_stats.skipped += (
                                1  # Folders returning None are skipped or failed
                            )

                        if current_count % 100 == 0:
                            logger.info(
                                f"Progress: {current_count}/{len(folders)} folders, "
                                f"{agg_stats.chunks_created} chunks"
                            )

                except Exception as exc:
                    logger.error("Future failed for %s: %s", folder.prefix, exc)
                    with stats_lock:
                        agg_stats.errors += 1

        logger.info(
            f"Batch complete: {agg_stats.folders_processed} folders processed, "
            f"{agg_stats.skipped} skipped, "
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
