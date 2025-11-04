"""
Chunking Service Module

Handles all text chunking operations for email processing and indexing.
"""

import logging
from pathlib import Path
from typing import Any

from emailops import llm_text_chunker as text_chunker

logger = logging.getLogger(__name__)


class ChunkingService:
    """Service for handling text chunking operations."""

    def __init__(self, export_root: str):
        """
        Initialize the chunking service.

        Args:
            export_root: Root directory for email exports
        """
        self.export_root = Path(export_root)
        self.chunks_dir = self.export_root / "_chunks"

    def force_rechunk_all(self) -> dict[str, Any]:
        """
        Force re-chunk all conversations using parallel workers.

        Note: Number of workers is determined by the config.

        Returns:
            Dictionary containing chunking results

        Raises:
            RuntimeError: If chunking operation fails
        """
        try:
            logger.info("Starting force rechunk all")

            result = text_chunker.force_rechunk_all(
                root=self.export_root,
            )

            # Process results for display
            processed_result = self._process_chunking_result(result)

            # Get num_workers from result for logging
            num_workers = processed_result.get("num_workers", "unknown")
            logger.info(
                f"Force rechunk completed: {processed_result['total_conversations']} conversations, "
                f"{processed_result['total_chunks']} chunks with {num_workers} workers"
            )

            return processed_result

        except Exception as e:
            logger.error(f"Force rechunk failed: {e}", exc_info=True)
            raise RuntimeError(f"Force rechunk operation failed: {e}") from e

    def incremental_chunk(self) -> dict[str, Any]:
        """
        Incrementally chunk only new/changed conversations.

        Note: Number of workers is determined by the config.

        Returns:
            Dictionary containing chunking results

        Raises:
            RuntimeError: If chunking operation fails
        """
        try:
            logger.info("Starting incremental chunk")

            result = text_chunker.incremental_rechunk(
                root=self.export_root,
            )

            # Process results for display
            processed_result = self._process_chunking_result(result)

            # Get num_workers from result for logging
            num_workers = processed_result.get("num_workers", "unknown")
            logger.info(
                f"Incremental chunk completed: {processed_result['total_conversations']} conversations, "
                f"{processed_result['total_chunks']} chunks with {num_workers} workers"
            )

            return processed_result

        except Exception as e:
            logger.error(f"Incremental chunk failed: {e}", exc_info=True)
            raise RuntimeError(f"Incremental chunk operation failed: {e}") from e

    def surgical_rechunk(self, conv_ids: list[str]) -> dict[str, Any]:
        """
        Re-chunk specific selected conversations.

        Args:
            conv_ids: List of conversation IDs to rechunk

        Returns:
            Dictionary containing chunking results

        Raises:
            ValueError: If conv_ids is empty
            RuntimeError: If chunking operation fails
        """
        if not conv_ids:
            raise ValueError("No conversation IDs provided for surgical rechunk")

        # Validate conversation IDs
        valid_ids = [cid for cid in conv_ids if cid and isinstance(cid, str)]
        if not valid_ids:
            raise ValueError("No valid conversation IDs provided")

        try:
            logger.info(f"Starting surgical rechunk for {len(valid_ids)} conversations")

            result = text_chunker.surgical_rechunk(
                root=self.export_root,
                conv_ids=valid_ids,
            )

            # Process results for display
            processed_result = self._process_chunking_result(result)

            # Get num_workers from result for logging
            num_workers = processed_result.get("num_workers", "unknown")
            logger.info(
                f"Surgical rechunk completed: {processed_result['total_conversations']} conversations, "
                f"{processed_result['total_chunks']} chunks with {num_workers} workers"
            )

            return processed_result

        except Exception as e:
            logger.error(f"Surgical rechunk failed: {e}", exc_info=True)
            raise RuntimeError(f"Surgical rechunk operation failed: {e}") from e

    def list_chunked_conversations(self) -> list[dict[str, Any]]:
        """
        List all conversations that have been chunked.

        Returns:
            List of chunked conversation information
        """
        import json
        from datetime import datetime

        if not self.chunks_dir.exists():
            logger.info("No chunks directory found")
            return []

        chunked_convs = []

        try:
            chunk_files = sorted(
                self.chunks_dir.glob("*.json"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )

            for chunk_file in chunk_files:
                try:
                    conv_id = chunk_file.stem
                    chunks = json.loads(chunk_file.read_text(encoding="utf-8"))
                    num_chunks = len(chunks)
                    last_mod = datetime.fromtimestamp(chunk_file.stat().st_mtime)

                    chunked_convs.append(
                        {
                            "conv_id": conv_id,
                            "num_chunks": num_chunks,
                            "status": "Chunked",
                            "last_modified": last_mod.strftime("%Y-%m-%d %H:%M:%S"),
                            "file_path": str(chunk_file),
                        }
                    )

                except Exception as e:
                    logger.warning(f"Could not process chunk file {chunk_file}: {e}")

            logger.info(f"Found {len(chunked_convs)} chunked conversations")
            return chunked_convs

        except Exception as e:
            logger.error(f"Failed to list chunked conversations: {e}", exc_info=True)
            return []

    def clear_chunks_directory(self) -> bool:
        """
        Clear all chunks from the chunks directory.

        Returns:
            True if successful, False otherwise
        """
        import shutil

        if not self.chunks_dir.exists():
            logger.info("No chunks directory to clear")
            return True

        try:
            shutil.rmtree(self.chunks_dir)
            logger.info(f"Cleared chunks directory: {self.chunks_dir}")
            return True

        except Exception as e:
            logger.error(f"Failed to clear chunks directory: {e}", exc_info=True)
            return False

    def get_chunk_statistics(self) -> dict[str, Any]:
        """
        Get statistics about chunked conversations.

        Returns:
            Dictionary containing chunk statistics
        """
        import json

        if not self.chunks_dir.exists():
            return {
                "total_conversations": 0,
                "total_chunks": 0,
                "avg_chunks_per_conv": 0,
                "total_size_mb": 0,
            }

        try:
            chunk_files = list(self.chunks_dir.glob("*.json"))
            total_convs = len(chunk_files)
            total_chunks = 0
            total_size = 0

            for chunk_file in chunk_files:
                try:
                    chunks = json.loads(chunk_file.read_text(encoding="utf-8"))
                    total_chunks += len(chunks)
                    total_size += chunk_file.stat().st_size
                except Exception:
                    pass

            avg_chunks = total_chunks / total_convs if total_convs > 0 else 0
            total_size_mb = total_size / (1024 * 1024)

            return {
                "total_conversations": total_convs,
                "total_chunks": total_chunks,
                "avg_chunks_per_conv": round(avg_chunks, 2),
                "total_size_mb": round(total_size_mb, 2),
            }

        except Exception as e:
            logger.error(f"Failed to get chunk statistics: {e}", exc_info=True)
            return {
                "total_conversations": 0,
                "total_chunks": 0,
                "avg_chunks_per_conv": 0,
                "total_size_mb": 0,
            }

    def _process_chunking_result(self, result: dict[str, Any]) -> dict[str, Any]:
        """
        Process raw chunking result for display.

        Args:
            result: Raw chunking result from text_chunker

        Returns:
            Processed result dictionary
        """
        # Extract conversation details from worker results
        conversation_details = []
        for worker_result in result.get("worker_results", []):
            for detail in worker_result.get("conversation_details", []):
                conversation_details.append(
                    {
                        "conv_id": detail["conv_id"],
                        "chunks": detail["num_chunks"],
                    }
                )

        # Compile summary statistics
        total_chunks = result.get("total_chunks", 0)
        total_convs = result.get("total_conversations", 0)
        failed_workers = result.get("failed_workers", [])

        return {
            "total_conversations": total_convs,
            "total_chunks": total_chunks,
            "conversation_details": conversation_details,
            "failed_workers": len(failed_workers),
            "success": len(failed_workers) == 0,
            "num_workers": result.get("num_workers", 0),
            "error_details": failed_workers if failed_workers else None,
        }

    def validate_export_root(self) -> tuple[bool, str]:
        """
        Validate the export root directory.

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not self.export_root.exists():
            return False, f"Export root does not exist: {self.export_root}"

        if not self.export_root.is_dir():
            return False, f"Export root is not a directory: {self.export_root}"

        # Check if there are any conversations to chunk
        conv_dirs = [
            d
            for d in self.export_root.iterdir()
            if d.is_dir() and not d.name.startswith("_") and not d.name.startswith(".")
        ]

        if not conv_dirs:
            return False, "No conversation directories found in export root"

        return True, ""
