"""
Indexing Service Module

Handles all index building and management operations.
"""

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class IndexingService:
    """Service for handling index building and management."""

    def __init__(self, export_root: str, index_dirname: str = ".email_index"):
        """
        Initialize the indexing service.

        Args:
            export_root: Root directory for email exports
            index_dirname: Name of the index directory
        """
        self.export_root = Path(export_root)
        self.index_dirname = index_dirname
        self.index_dir = self.export_root / index_dirname

    def build_index(
        self,
        provider: str = "vertex",
        batch_size: int = 64,
        num_workers: int = 4,
        force: bool = False,
        limit: int | None = None,
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> dict[str, Any]:
        """
        Build or update the index - REWRITTEN TO CALL BACKEND DIRECTLY.

        Args:
            provider: Embedding provider
            batch_size: Batch size for processing
            num_workers: Number of parallel workers
            force: Force full re-index
            limit: Limit per conversation (0 or None for unlimited)
            progress_callback: Callback(current, total, message) for progress updates

        Returns:
            Dictionary containing build results

        Raises:
            ValueError: If export root is invalid
            RuntimeError: If index build fails
        """
        if not self.export_root.exists():
            raise ValueError(f"Export root does not exist: {self.export_root}")

        logger.info(f"Starting index build: provider={provider}, workers={num_workers}, batch={batch_size}")

        try:
            # Import backend indexing functions directly
            from emailops.core_config import get_config
            from emailops.indexing_main import (
                build_corpus,
                build_incremental_corpus,
                load_existing_index,
                save_index,
            )

            # Ensure export root and index directory exist
            self.index_dir.mkdir(parents=True, exist_ok=True)

            # Update config with parameters
            config = get_config()
            config.core.provider = provider
            config.processing.batch_size = batch_size
            config.processing.num_workers = num_workers
            config.update_environment()

            # Load existing index if not forcing rebuild
            _, existing_mapping, existing_file_times, _existing_embeddings = load_existing_index(self.index_dir)

            # Build corpus
            if force or not existing_file_times:
                logger.info("Building full corpus (force or no existing index)")
                new_docs, unchanged_docs = build_corpus(
                    root=self.export_root,
                    index_dir=self.index_dir,
                    last_run_time=None,
                    limit=limit
                )
            else:
                logger.info("Building incremental corpus")
                new_docs, deleted_ids = build_incremental_corpus(
                    root=self.export_root,
                    existing_file_times=existing_file_times,
                    existing_mapping=existing_mapping or [],
                    limit=limit
                )
                # Filter out deleted docs from unchanged
                unchanged_docs = [
                    d for d in (existing_mapping or [])
                    if d.get("id") not in deleted_ids and d.get("id") not in {nd["id"] for nd in new_docs}
                ]

            total_docs = len(new_docs) + len(unchanged_docs)
            logger.info(f"Corpus built: {len(new_docs)} new/updated, {len(unchanged_docs)} unchanged, {total_docs} total")

            # Report progress
            if progress_callback:
                try:
                    progress_callback(1, 5, f"Corpus built: {total_docs} documents")
                except Exception as e:
                    logger.warning(f"Progress callback error: {e}")

            # Handle empty corpus
            if not total_docs:
                logger.warning("No documents to index")
                return {
                    "success": True,
                    "num_documents": 0,
                    "num_conversations": 0,
                    "index_dir": str(self.index_dir)
                }

            # Materialize text for documents that need embedding
            if progress_callback:
                try:
                    progress_callback(2, 5, "Preparing documents for embedding...")
                except Exception as e:
                    logger.warning(f"Progress callback error: {e}")

            from emailops.indexing_main import _materialize_text_for_docs
            from emailops.services.file_service import FileService
            all_docs = new_docs + unchanged_docs
            file_service = FileService(export_root=str(self.export_root))
            materialized_docs = _materialize_text_for_docs(all_docs, file_service)
            valid_docs = [d for d in materialized_docs if str(d.get("text", "")).strip()]

            logger.info(f"Materialized {len(valid_docs)} valid documents")

            # Embed documents in batches
            if progress_callback:
                try:
                    progress_callback(3, 5, f"Embedding {len(valid_docs)} documents...")
                except Exception as e:
                    logger.warning(f"Progress callback error: {e}")

            import numpy as np

            from emailops.llm_client_shim import embed_texts

            all_embeddings: list[np.ndarray] = []
            texts = [str(d["text"]) for d in valid_docs]

            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i : i + batch_size]
                logger.debug(f"Embedding batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")

                try:
                    batch_vecs = embed_texts(batch_texts, provider=provider)
                    batch_vecs = np.asarray(batch_vecs, dtype="float32")

                    # Validate batch
                    if batch_vecs.ndim != 2 or batch_vecs.shape[0] != len(batch_texts):
                        raise RuntimeError(f"Invalid embeddings shape: {batch_vecs.shape} for {len(batch_texts)} texts")

                    all_embeddings.append(batch_vecs)

                    # Progress update per batch
                    if progress_callback and len(texts) > batch_size:
                        try:
                            progress_callback(
                                min(i + batch_size, len(texts)),
                                len(texts),
                                f"Embedded {min(i + batch_size, len(texts))}/{len(texts)} documents"
                            )
                        except Exception as e:
                            logger.warning(f"Progress callback error: {e}")

                except Exception as e:
                    logger.error(f"Embedding batch {i}:{i+batch_size} failed: {e}")
                    raise

            # Stack all embeddings
            embeddings = np.vstack(all_embeddings)
            logger.info(f"Created embeddings array: shape={embeddings.shape}")

            # Save index
            if progress_callback:
                try:
                    progress_callback(4, 5, "Saving index...")
                except Exception as e:
                    logger.warning(f"Progress callback error: {e}")

            # Prepare mapping for save
            mapping_out = []
            for d in valid_docs:
                text = str(d.get("text", "") or "")
                snippet = text[:500] if text else ""

                rec = {
                    "id": d.get("id"),
                    "path": d.get("path"),
                    "conv_id": d.get("conv_id"),
                    "doc_type": d.get("doc_type"),
                    "subject": d.get("subject", ""),
                    "date": d.get("date"),
                    "snippet": snippet,
                }
                mapping_out.append(rec)

            # Save index artifacts
            num_convs = len({d.get("conv_id") for d in valid_docs if d.get("conv_id")})
            save_index(
                index_dir=self.index_dir,
                embeddings=embeddings,
                mapping=mapping_out,
                provider=provider,
                num_folders=num_convs
            )

            if progress_callback:
                try:
                    progress_callback(5, 5, "Index build complete")
                except Exception as e:
                    logger.warning(f"Progress callback error: {e}")

            result = {
                "success": True,
                "num_documents": len(valid_docs),
                "num_conversations": num_convs,
                "index_dir": str(self.index_dir),
                "new_documents": len(new_docs),
                "unchanged_documents": len(unchanged_docs),
                "embedding_dimensions": int(embeddings.shape[1])
            }

            logger.info(f"Index build completed: {result}")
            return result

        except Exception as e:
            logger.error(f"Index build failed: {e}", exc_info=True)
            raise RuntimeError(f"Index build failed: {e}") from e

    def validate_index(self) -> tuple[bool, str]:
        """
        Validate that the index exists and is ready - FIXED validation logic.

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not self.index_dir.exists():
            return False, f"Index directory not found at {self.index_dir}"

        # Check for required index files
        mapping_file = self.index_dir / "mapping.json"
        if not mapping_file.exists():
            return False, "Index mapping.json not found"

        # Check for embeddings.npy file (NOT directory)
        embeddings_file = self.index_dir / "embeddings.npy"
        if not embeddings_file.exists():
            return False, "Embeddings file (embeddings.npy) not found"

        # Optional: Check for meta.json
        meta_file = self.index_dir / "meta.json"
        if not meta_file.exists():
            logger.warning("meta.json not found in index directory")

        return True, ""

    def get_index_statistics(self) -> dict[str, Any]:
        """
        Get statistics about the current index - FIXED mapping iteration.

        Returns:
            Dictionary containing index statistics
        """
        import json

        stats = {
            "exists": False,
            "num_documents": 0,
            "num_conversations": 0,
            "total_size_mb": 0,
            "last_updated": None,
        }

        if not self.index_dir.exists():
            return stats

        stats["exists"] = True

        # Read mapping file for document count
        mapping_file = self.index_dir / "mapping.json"
        if mapping_file.exists():
            try:
                with mapping_file.open("r", encoding="utf-8") as f:
                    mapping = json.load(f)

                    # Mapping is a list of dicts, not a dict
                    if isinstance(mapping, list):
                        stats["num_documents"] = len(mapping)

                        # Count unique conversations from doc dicts
                        conv_ids = set()
                        for doc in mapping:
                            if isinstance(doc, dict):
                                # Get doc_id from dict and extract conv_id
                                doc_id = doc.get("id", "")
                                if doc_id and "::" in doc_id:
                                    base_id = doc_id.split("::")[0]
                                    conv_ids.add(base_id)
                                elif doc.get("conv_id"):
                                    conv_ids.add(doc["conv_id"])
                        stats["num_conversations"] = len(conv_ids)

                        # Get last modified time
                        import datetime

                        mtime = mapping_file.stat().st_mtime
                        stats["last_updated"] = datetime.datetime.fromtimestamp(
                            mtime
                        ).isoformat()
                    else:
                        logger.warning(f"Mapping file has unexpected type: {type(mapping)}")

            except Exception as e:
                logger.warning(f"Failed to read mapping file: {e}")

        # Calculate total size
        total_size = 0
        for item in self.index_dir.rglob("*"):
            if item.is_file():
                total_size += item.stat().st_size

        stats["total_size_mb"] = round(total_size / (1024 * 1024), 2)

        return stats

    def clear_index(self) -> bool:
        """
        Clear the entire index directory.

        Returns:
            True if successful, False otherwise
        """
        import shutil

        if not self.index_dir.exists():
            logger.info("No index directory to clear")
            return True

        try:
            shutil.rmtree(self.index_dir)
            logger.info(f"Cleared index directory: {self.index_dir}")
            return True
        except Exception as e:
            logger.error(f"Failed to clear index: {e}", exc_info=True)
            return False

    def check_dependencies(self) -> dict[str, Any]:
        """
        Check if required dependencies for indexing are installed - FIXED.

        Returns:
            Dictionary with dependency status
        """
        dependencies = {
            "numpy": False,
            "vertexai": False,
            "google-cloud-aiplatform": False,
            "all_satisfied": False,
        }

        try:
            import importlib.util
            if importlib.util.find_spec("numpy") is not None:
                dependencies["numpy"] = True
        except ImportError:
            pass

        try:
            import importlib.util
            if importlib.util.find_spec("vertexai") is not None:
                dependencies["vertexai"] = True
        except ImportError:
            pass

        try:
            import importlib.util
            if importlib.util.find_spec("google.cloud.aiplatform") is not None:
                dependencies["google-cloud-aiplatform"] = True
        except ImportError:
            pass

        dependencies["all_satisfied"] = all(
            [
                dependencies["numpy"],
                dependencies["vertexai"],
                dependencies["google-cloud-aiplatform"],
            ]
        )

        return dependencies

    def estimate_indexing_time(self, num_conversations: int) -> dict[str, Any]:
        """
        Estimate time required for indexing.

        Args:
            num_conversations: Number of conversations to index

        Returns:
            Dictionary with time estimates
        """
        # Rough estimates based on typical performance
        avg_conv_time_seconds = 2.5  # Average time per conversation

        total_seconds = num_conversations * avg_conv_time_seconds

        return {
            "num_conversations": num_conversations,
            "estimated_seconds": int(total_seconds),
            "estimated_minutes": round(total_seconds / 60, 1),
            "estimated_hours": (
                round(total_seconds / 3600, 2) if total_seconds > 3600 else 0
            ),
        }

    def get_indexing_config(self) -> dict[str, Any]:
        """
        Get current indexing configuration.

        Returns:
            Dictionary with configuration values
        """
        from emailops.core_config import get_config

        config = get_config()

        return {
            "export_root": str(self.export_root),
            "index_dirname": self.index_dirname,
            "provider": config.core.provider,
            "chunk_size": config.processing.chunk_size,
            "chunk_overlap": config.processing.chunk_overlap,
            "num_workers": config.processing.num_workers,
            "batch_size": config.processing.batch_size,
        }
