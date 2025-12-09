"""
CLI wrapper/orchestrator around mailroom.process_job.
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from cortex.config.loader import get_config
from cortex.db.models import IngestJob
from cortex.db.session import SessionLocal, set_session_tenant
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

    def _persist_job(
        self,
        job_request: IngestJobRequest,
        *,
        status: str,
        stats: Dict[str, Any] | None = None,
        metadata: Dict[str, Any] | None = None,
    ) -> None:
        with SessionLocal() as session:
            set_session_tenant(session, job_request.tenant_id)
            job = IngestJob(
                job_id=job_request.job_id,
                tenant_id=job_request.tenant_id,
                source_type=job_request.source_type,
                source_uri=job_request.source_uri,
                status=status,
                options=job_request.options,
                stats=stats or {},
                metadata_=metadata or {},
            )
            session.merge(job)
            session.commit()

    def _update_job(
        self, job_request: IngestJobRequest, summary: IngestJobSummary
    ) -> None:
        status = _derive_status(summary)
        stats = summary.model_dump()
        metadata = {
            "prefix": job_request.source_uri,
            "problems": [p.model_dump() for p in summary.problems],
            "aborted_reason": summary.aborted_reason,
        }
        self._persist_job(job_request, status=status, stats=stats, metadata=metadata)

    def _build_job_request(self, folder_prefix: str) -> IngestJobRequest:
        return IngestJobRequest(
            job_id=uuid.uuid4(),
            tenant_id=self.tenant_id,
            source_type="s3",
            source_uri=folder_prefix,
            options={"prefix": folder_prefix},
        )

    def process_folder(self, folder_prefix: str) -> Optional[IngestJobSummary]:
        job_request = self._build_job_request(folder_prefix)
        try:
            self._persist_job(
                job_request,
                status="pending",
                metadata={"prefix": folder_prefix},
            )
            summary = process_job(job_request)
            self._update_job(job_request, summary)
            self.stats.folders_processed += 1
            self.stats.jobs_created += 1
            return summary
        except Exception as exc:  # broad catch to keep orchestrator resilient
            logger.error("Ingestion failed for %s: %s", folder_prefix, exc)
            self.stats.errors += 1
            try:
                self._persist_job(
                    job_request,
                    status="failed",
                    stats={"error": str(exc)},
                    metadata={"prefix": folder_prefix},
                )
            except Exception:
                logger.exception("Failed to record job failure for %s", folder_prefix)
            return None

    def run_full_ingestion(
        self, prefix: str = "raw/outlook/", limit: Optional[int] = None
    ) -> List[IngestJobSummary]:
        logger.info("Starting ingestion scan for prefix %s", prefix)
        summaries: List[IngestJobSummary] = []

        folders = list(
            self.s3_handler.list_conversation_folders(prefix=prefix, limit=limit)
        )
        for folder in folders:
            summary = self.process_folder(folder.prefix)
            if summary:
                summaries.append(summary)

        return summaries


def run_ingestion_cli() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run S3 ingestion pipeline")
    parser.add_argument("--prefix", default="raw/outlook/", help="S3 prefix")
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
