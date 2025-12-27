"""
Unified Pipeline Orchestrator.

Implements the "Atomic Batch" processing logic defined in the Unified Pipeline Design.
Orchestrates: Source -> Ingest -> Embed
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Optional
from uuid import UUID

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
        self, source_prefix: str = "Outlook/", limit: Optional[int] = None
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

        # 2. Parallel Processing
        logger.info(
            f"Pipeline: Starting parallel processing (concurrency={self.concurrency})"
        )

        with ThreadPoolExecutor(max_workers=self.concurrency) as executor:
            future_to_folder = {}
            for folder in folders_iter:
                self.stats.folders_found += 1
                future = executor.submit(self._process_single_folder, folder)
                future_to_folder[future] = folder

            for future in as_completed(future_to_folder):
                folder = future_to_folder[future]
                try:
                    future.result()
                except Exception as e:
                    logger.exception(f"Pipeline: Error processing folder '{folder}': {e}")

        self.indexer.shutdown()
        total_time = self.stats.duration_seconds
        logger.info(
            f"Pipeline complete in {total_time:.2f}s. "
            f"Processed: {self.stats.folders_processed}, "
            f"Skipped: {self.stats.folders_skipped}, "
            f"Failed: {self.stats.folders_failed}"
        )
        return self.stats

    def _process_single_folder(self, folder: str) -> None:
        """
        Process a single conversation folder atomically.
        """
        folder_name = (
            getattr(folder, "name", None)
            or getattr(folder, "prefix", None)
            or str(folder)
        )

        # Dry-run mode: log and skip
        if self.dry_run:
            logger.info(f"[DRY-RUN] Would process: {folder_name}")
            self.stats.folders_processed += 1
            return

        try:
            # Step A: Ingestion
            # IngestionProcessor.process_folder handles idempotency check internally
            summary = self.processor.process_folder(folder)

            if summary is None:
                # Skipped due to idempotency (unchanged)
                self.stats.folders_skipped += 1
                return

            if summary.aborted_reason or summary.problems:
                self.stats.folders_failed += 1
                self.stats.errors_list.append(
                    {
                        "folder": folder_name,
                        "reason": summary.aborted_reason or str(summary.problems),
                    }
                )
                logger.error(
                    f"Ingestion failed for {folder_name}: {summary.aborted_reason}"
                )
                return

            # Success
            self.stats.folders_processed += 1
            self.stats.chunks_created += summary.chunks_created

            # Step B: Auto-Embedding
            conv_id = getattr(summary, "conversation_id", None)
            if self.auto_embed and conv_id:
                self._trigger_embedding(conv_id, summary.chunks_created)

        except Exception as e:
            self.stats.folders_failed += 1
            self.stats.errors_list.append({"folder": folder_name, "reason": str(e)})
            logger.exception(f"Pipeline error processing {folder_name}: {e}")

    def _trigger_embedding(self, conversation_id: UUID, chunk_count: int) -> None:
        """
        Trigger embedding generation for a conversation.
        """
        if chunk_count == 0:
            return

        try:
            logger.info(
                f"Auto-embedding conversation {conversation_id} ({chunk_count} chunks)"
            )

            # NOTE: Ideally we would call a specialized indexer here.
            # For now, we reuse the pattern from parallel_indexer but scoped to one conversation.
            # Since we don't have a granular "index_one_conversation" method exposed yet,
            # we will mark this as a TODO for the next iteration or require `cortex index` to be runnable.

            # For this MVP, we will count it as "queued" conceptually.
            # In a real impl, we would call:
            # indexer.index_conversation(conversation_id)
            self.indexer.enqueue_conversation(conversation_id)
            self.stats.embeddings_generated += chunk_count
        except Exception as e:
            logger.error(f"Embedding failure for {conversation_id}: {e}")
