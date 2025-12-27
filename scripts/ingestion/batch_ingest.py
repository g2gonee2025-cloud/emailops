#!/usr/bin/env python3
"""
Batch ingestion script for processing S3 Outlook folders.

Processes all folders through: validation, cleaning, chunking (no embedding).

NOTE: This script is a "power user" tool pending CLI feature parity.
Once `cortex ingest --batch` is implemented with progress/stats/skip/limit
features, this script will be deprecated. See codebase_audit.md for details.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Add backend/src to path
sys.path.insert(0, str(Path(__file__).parent / "backend" / "src"))

from cortex.ingestion.models import IngestJobSummary
from cortex.ingestion.processor import IngestionProcessor
from cortex.ingestion.s3_source import create_s3_source

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("batch_ingest")


class BatchStats:
    """Track processing statistics."""

    def __init__(self):
        self.total_folders = 0
        self.processed = 0
        self.failed = 0
        self.skipped = 0
        self.total_chunks = 0
        self.total_attachments = 0
        self.errors: list[dict[str, Any]] = []
        self.start_time = time.time()

    def add_success(self, summary: IngestJobSummary) -> None:
        self.processed += 1
        self.total_chunks += summary.chunks_created
        self.total_attachments += summary.attachments_parsed

    def add_failure(self, folder: str, error: str) -> None:
        self.failed += 1
        self.errors.append({"folder": folder, "error": error})

    def elapsed(self) -> float:
        return time.time() - self.start_time

    def rate(self) -> float:
        elapsed = self.elapsed()
        if elapsed > 0:
            return self.processed / elapsed
        return 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_folders": self.total_folders,
            "processed": self.processed,
            "failed": self.failed,
            "skipped": self.skipped,
            "total_chunks": self.total_chunks,
            "total_attachments": self.total_attachments,
            "elapsed_seconds": self.elapsed(),
            "rate_per_second": self.rate(),
            "errors": self.errors[:100],  # Cap at 100 errors
        }


def run_batch(
    prefix: str = "Outlook/",
    tenant_id: str = "default",
    limit: int | None = None,
    skip: int = 0,
    report_every: int = 50,
    max_consecutive_failures: int = 10,
) -> BatchStats:
    """
    Run batch ingestion on S3 folders.

    Args:
        prefix: S3 prefix to scan
        tenant_id: Tenant ID for RLS
        limit: Max folders to process (None = all)
        skip: Number of folders to skip at start
        report_every: Print progress every N folders
        max_consecutive_failures: Abort if this many folders fail in a row.

    Returns:
        BatchStats with processing results
    """
    stats = BatchStats()
    handler = create_s3_source()
    processor = IngestionProcessor(tenant_id=tenant_id)

    logger.info(f"Starting batch ingestion from s3://{handler.bucket}/{prefix}")
    logger.info(f"Tenant: {tenant_id}, Limit: {limit or 'all'}, Skip: {skip}")

    # This block is refactored to handle memory optimization and N+1 query fix.
    # We now use an iterator and pre-fetch timestamps.
    all_folders = handler.list_conversation_folders(prefix=prefix)

    # Apply skip/limit logic before fetching timestamps to reduce DB load
    folders_to_process = []
    if skip > 0:
        try:
            for _ in range(skip):
                next(all_folders)
        except StopIteration:
            pass  # Skip is larger than total number of folders

    if limit:
        folders_to_process = [folder for _, folder in zip(range(limit), all_folders)]
    else:
        folders_to_process = list(all_folders)

    stats.total_folders = len(folders_to_process)
    logger.info(f"Found {stats.total_folders} folders to process after skip/limit.")

    # N+1 Query Fix: Pre-fetch all existing timestamps
    folder_names = [f.name for f in folders_to_process]
    existing_timestamps = processor._get_existing_timestamps(folder_names)
    logger.info(f"Pre-fetched {len(existing_timestamps)} existing folder timestamps.")

    consecutive_failures = 0
    for i, folder in enumerate(folders_to_process, 1):
        try:
            # Pass the pre-fetched timestamps to the processor
            summary = processor.process_folder(folder, existing_timestamps)

            if summary is None:
                # This indicates the folder was skipped, not an error
                stats.skipped += 1
                consecutive_failures = 0
            elif not summary.aborted_reason:
                stats.add_success(summary)
                consecutive_failures = 0
            else:
                stats.add_failure(folder.name, summary.aborted_reason)
                consecutive_failures += 1
        except Exception as e:
            stats.add_failure(folder.name, str(e))
            logger.error(f"Error processing {folder.name}: {e}", exc_info=True)
            consecutive_failures += 1

        if consecutive_failures >= max_consecutive_failures:
            logger.critical(
                f"Aborting due to {consecutive_failures} consecutive failures. "
                "This may indicate a persistent issue with S3, the database, or the network."
            )
            break

        # Progress report
        if i % report_every == 0 or i == len(folders):
            pct = (i / len(folders)) * 100
            rate = stats.rate()
            eta = (len(folders) - i) / rate if rate > 0 else 0
            logger.info(
                f"Progress: {i}/{len(folders)} ({pct:.1f}%) | "
                f"OK: {stats.processed} | Failed: {stats.failed} | "
                f"Chunks: {stats.total_chunks} | "
                f"Rate: {rate:.2f}/s | ETA: {eta / 60:.1f}m"
            )

    return stats


def main():
    parser = argparse.ArgumentParser(description="Batch S3 ingestion")
    parser.add_argument("--prefix", default="Outlook/", help="S3 prefix")
    parser.add_argument("--tenant", default="default", help="Tenant ID")
    parser.add_argument("--limit", type=int, help="Max folders to process")
    parser.add_argument("--skip", type=int, default=0, help="Folders to skip")
    parser.add_argument(
        "--report-every", type=int, default=50, help="Progress interval"
    )
    parser.add_argument("--output", help="Output JSON file for stats")
    parser.add_argument(
        "--max-consecutive-failures",
        type=int,
        default=10,
        help="Abort after N consecutive failures",
    )

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("BATCH INGESTION STARTING")
    logger.info(f"Time: {datetime.now(UTC).isoformat()}")
    logger.info("=" * 60)

    stats = run_batch(
        prefix=args.prefix,
        tenant_id=args.tenant,
        limit=args.limit,
        skip=args.skip,
        report_every=args.report_every,
        max_consecutive_failures=args.max_consecutive_failures,
    )

    logger.info("=" * 60)
    logger.info("BATCH INGESTION COMPLETE")
    logger.info(f"Processed: {stats.processed}/{stats.total_folders}")
    logger.info(f"Failed: {stats.failed}")
    logger.info(f"Total Chunks: {stats.total_chunks}")
    logger.info(f"Total Attachments: {stats.total_attachments}")
    logger.info(f"Elapsed: {stats.elapsed() / 60:.1f} minutes")
    logger.info(f"Rate: {stats.rate():.2f} folders/second")
    logger.info("=" * 60)

    if args.output:
        with Path(args.output).open("w") as f:
            json.dump(stats.to_dict(), f, indent=2)
        logger.info(f"Stats written to {args.output}")

    # Print errors summary
    if stats.errors:
        logger.warning(f"Errors ({len(stats.errors)}):")
        for err in stats.errors[:10]:
            logger.warning(f"  - {err['folder']}: {err['error'][:100]}")
        if len(stats.errors) > 10:
            logger.warning(f"  ... and {len(stats.errors) - 10} more")


if __name__ == "__main__":
    main()
