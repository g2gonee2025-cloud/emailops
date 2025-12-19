#!/usr/bin/env python3
"""
Batch ingestion script for processing S3 Outlook folders.

Processes all folders through: validation, cleaning, chunking (no embedding).
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
) -> BatchStats:
    """
    Run batch ingestion on S3 folders.

    Args:
        prefix: S3 prefix to scan
        tenant_id: Tenant ID for RLS
        limit: Max folders to process (None = all)
        skip: Number of folders to skip at start
        report_every: Print progress every N folders

    Returns:
        BatchStats with processing results
    """
    stats = BatchStats()
    handler = create_s3_source()
    processor = IngestionProcessor(tenant_id=tenant_id)

    logger.info(f"Starting batch ingestion from s3://{handler.bucket}/{prefix}")
    logger.info(f"Tenant: {tenant_id}, Limit: {limit or 'all'}, Skip: {skip}")

    # List all folders
    folders = list(handler.list_conversation_folders(prefix=prefix, limit=None))
    stats.total_folders = len(folders)
    logger.info(f"Found {stats.total_folders} folders")

    # Apply skip/limit
    if skip > 0:
        folders = folders[skip:]
        logger.info(f"Skipping first {skip} folders, {len(folders)} remaining")

    if limit:
        folders = folders[:limit]
        logger.info(f"Limited to {limit} folders")

    for i, folder in enumerate(folders, 1):
        try:
            summary = processor.process_folder(folder.prefix)

            if summary and not summary.aborted_reason:
                stats.add_success(summary)
            elif summary and summary.aborted_reason:
                stats.add_failure(folder.name, summary.aborted_reason)
            else:
                stats.add_failure(folder.name, "No summary returned")

        except Exception as e:
            stats.add_failure(folder.name, str(e))
            logger.error(f"Error processing {folder.name}: {e}")

        # Progress report
        if i % report_every == 0 or i == len(folders):
            pct = (i / len(folders)) * 100
            rate = stats.rate()
            eta = (len(folders) - i) / rate if rate > 0 else 0
            logger.info(
                f"Progress: {i}/{len(folders)} ({pct:.1f}%) | "
                f"OK: {stats.processed} | Failed: {stats.failed} | "
                f"Chunks: {stats.total_chunks} | "
                f"Rate: {rate:.2f}/s | ETA: {eta/60:.1f}m"
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
    )

    logger.info("=" * 60)
    logger.info("BATCH INGESTION COMPLETE")
    logger.info(f"Processed: {stats.processed}/{stats.total_folders}")
    logger.info(f"Failed: {stats.failed}")
    logger.info(f"Total Chunks: {stats.total_chunks}")
    logger.info(f"Total Attachments: {stats.total_attachments}")
    logger.info(f"Elapsed: {stats.elapsed()/60:.1f} minutes")
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
