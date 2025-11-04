"""
Batch Service Module

Handles batch operations for summarization and email drafting.
"""

import asyncio
import functools
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from emailops import feature_summarize as summarizer
from emailops.feature_search_draft import draft_email_reply_eml

from .base_service import BaseService

logger = logging.getLogger(__name__)


class BatchService(BaseService):
    """Service for handling batch operations."""

    def __init__(self, export_root: str):
        """
        Initialize the batch service.

        Args:
            export_root: Root directory for email exports
        """
        super().__init__(export_root)

    async def batch_summarize(
        self,
        conv_ids: list[str],
        output_dir: Path,
        temperature: float = 0.7,
        merge_manifest: bool = True,
        progress_callback: Callable[[int, int, str, str], None] | None = None,
    ) -> dict[str, Any]:
        if not conv_ids:
            raise ValueError("No conversation IDs provided for batch summarization")

        output_dir.mkdir(parents=True, exist_ok=True)

        async def _summarize_one(conv_id: str, i: int):
            try:
                if progress_callback:
                    progress_callback(i, len(conv_ids), conv_id, "Processing")

                conv_path = self.file_service.get_conversation_path(conv_id)
                if not conv_path:
                    raise ValueError(f"Invalid conversation ID: {conv_id}")

                logger.info(f"Summarizing conversation {i}/{len(conv_ids)}: {conv_id}")
                analysis = await summarizer.analyze_conversation_dir(
                    thread_dir=conv_path,
                    temperature=temperature,
                    merge_manifest=merge_manifest,
                )

                json_path = output_dir / f"{conv_id}_summary.json"
                self.file_service.save_json(analysis, json_path)

                if progress_callback:
                    progress_callback(i, len(conv_ids), conv_id, "Completed")

                return {
                    "conv_id": conv_id,
                    "output_file": str(json_path),
                    "summary": analysis.get("brief_summary", ""),
                }
            except Exception as e:
                logger.error(f"Failed to summarize {conv_id}: {e}")
                if progress_callback:
                    progress_callback(i, len(conv_ids), conv_id, "Failed")
                return {"conv_id": conv_id, "error": str(e)}

        tasks = [_summarize_one(conv_id, i) for i, conv_id in enumerate(conv_ids, 1)]
        task_results = await asyncio.gather(*tasks)

        results = {
            "operation": "summarize",
            "total": len(conv_ids),
            "completed": 0,
            "failed": 0,
            "summaries": [],
            "errors": [],
        }
        for res in task_results:
            if "error" in res:
                results["failed"] += 1
                results["errors"].append(res)
            else:
                results["completed"] += 1
                results["summaries"].append(res)

        logger.info(
            f"Batch summarization complete: {results['completed']} succeeded, "
            f"{results['failed']} failed out of {results['total']}"
        )
        return results

    async def batch_draft_replies(
        self,
        conv_ids: list[str],
        output_dir: Path,
        provider: str = "vertex",
        sim_threshold: float = 0.5,
        target_tokens: int = 20000,
        temperature: float = 0.7,
        reply_policy: str = "reply_all",
        include_attachments: bool = True,
        progress_callback: Callable[[int, int, str, str], None] | None = None,
    ) -> dict[str, Any]:
        if not conv_ids:
            raise ValueError("No conversation IDs provided for batch drafting")

        output_dir.mkdir(parents=True, exist_ok=True)

        async def _draft_one(conv_id: str, i: int):
            try:
                if progress_callback:
                    progress_callback(i, len(conv_ids), conv_id, "Processing")

                conv_path = self.file_service.get_conversation_path(conv_id)
                if not conv_path:
                    raise ValueError(f"Invalid conversation ID: {conv_id}")

                logger.info(f"Drafting reply {i}/{len(conv_ids)}: {conv_id}")
                # This function is not async, so we run it in a thread
                loop = asyncio.get_running_loop()
                func = functools.partial(
                    draft_email_reply_eml,
                    export_root=self.export_root,
                    conv_id=conv_id,
                    query="",
                    provider=provider,
                    sim_threshold=sim_threshold,
                    target_tokens=target_tokens,
                    temperature=temperature,
                    reply_policy=reply_policy,
                    include_attachments=include_attachments,
                )
                result = await loop.run_in_executor(None, func)

                eml_path = output_dir / f"{conv_id}_reply.eml"
                self.file_service.save_binary_file(result["eml_bytes"], eml_path)

                if progress_callback:
                    progress_callback(i, len(conv_ids), conv_id, "Completed")

                return {
                    "conv_id": conv_id,
                    "output_file": str(eml_path),
                    "subject": result.get("subject", ""),
                    "preview": result.get("body_preview", "")[:200],
                }
            except Exception as e:
                logger.error(f"Failed to draft reply for {conv_id}: {e}")
                if progress_callback:
                    progress_callback(i, len(conv_ids), conv_id, "Failed")
                return {"conv_id": conv_id, "error": str(e)}

        tasks = [_draft_one(conv_id, i) for i, conv_id in enumerate(conv_ids, 1)]
        task_results = await asyncio.gather(*tasks)

        results = {
            "operation": "draft_replies",
            "total": len(conv_ids),
            "completed": 0,
            "failed": 0,
            "drafts": [],
            "errors": [],
        }
        for res in task_results:
            if "error" in res:
                results["failed"] += 1
                results["errors"].append(res)
            else:
                results["completed"] += 1
                results["drafts"].append(res)

        logger.info(
            f"Batch draft replies complete: {results['completed']} succeeded, "
            f"{results['failed']} failed out of {results['total']}"
        )
        return results

    def estimate_batch_time(
        self, num_items: int, operation: str = "summarize"
    ) -> dict[str, Any]:
        """
        Estimate time required for batch operation.

        Args:
            num_items: Number of items to process
            operation: Type of operation ("summarize" or "draft_replies")

        Returns:
            Dictionary with time estimates
        """
        # Rough estimates based on typical performance
        time_per_item = {
            "summarize": 15,  # seconds
            "draft_replies": 20,  # seconds
        }

        seconds_per_item = time_per_item.get(operation, 15)
        total_seconds = num_items * seconds_per_item

        return {
            "operation": operation,
            "num_items": num_items,
            "estimated_seconds": total_seconds,
            "estimated_minutes": round(total_seconds / 60, 1),
            "estimated_hours": (
                round(total_seconds / 3600, 2) if total_seconds > 3600 else 0
            ),
        }

    def get_batch_statistics(self, results: dict[str, Any]) -> dict[str, Any]:
        """
        Extract statistics from batch results.

        Args:
            results: Batch operation results

        Returns:
            Dictionary containing statistics
        """
        stats = {
            "success_rate": 0.0,
            "failure_rate": 0.0,
            "total_processed": results.get("completed", 0) + results.get("failed", 0),
            "completed": results.get("completed", 0),
            "failed": results.get("failed", 0),
            "errors_summary": [],
        }

        if stats["total_processed"] > 0:
            stats["success_rate"] = round(
                (stats["completed"] / stats["total_processed"]) * 100, 2
            )
            stats["failure_rate"] = round(
                (stats["failed"] / stats["total_processed"]) * 100, 2
            )

        # Summarize errors
        errors = results.get("errors", [])
        if errors:
            error_types = {}
            for err in errors:
                error_msg = str(err.get("error", "Unknown error"))
                # Extract error type from message
                error_type = error_msg.split(":")[0] if ":" in error_msg else "Unknown"
                error_types[error_type] = error_types.get(error_type, 0) + 1

            stats["errors_summary"] = [
                {"type": k, "count": v} for k, v in error_types.items()
            ]

        return stats

    def validate_batch_items(self, conv_ids: list[str]) -> tuple[list[str], list[str]]:
        """
        Validate batch items before processing.

        Args:
            conv_ids: List of conversation IDs

        Returns:
            Tuple of (valid_ids, invalid_ids)
        """
        valid = []
        invalid = []

        for conv_id in conv_ids:
            if self.file_service.get_conversation_path(conv_id):
                valid.append(conv_id)
            else:
                invalid.append(conv_id)

        return valid, invalid
