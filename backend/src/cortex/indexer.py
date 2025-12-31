"""
Unified Indexer for Embeddings.
"""

from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from uuid import UUID

from cortex.common.exceptions import ConfigurationError
from cortex.config.loader import get_config
from cortex.embeddings.client import EmbeddingsClient
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

logger = logging.getLogger(__name__)

MAX_INFLIGHT_MULTIPLIER = 4
DEFAULT_EMBED_BATCH_SIZE = 256


class Indexer:
    """
    Handles embedding and indexing of conversation chunks.
    """

    def __init__(self, concurrency: int = 4):
        self.concurrency = max(1, concurrency)
        self.executor = ThreadPoolExecutor(max_workers=self.concurrency)
        self.embedding_client = EmbeddingsClient()

        config = get_config()
        db_url = getattr(config.database, "url", None)
        if not db_url:
            raise ConfigurationError(
                "Database URL is required for indexer",
                error_code="DB_URL_MISSING",
            )
        self.engine = create_engine(db_url)
        self.Session = sessionmaker(bind=self.engine)

        embed_batch_size = getattr(
            getattr(config, "embedding", None),
            "batch_size",
            DEFAULT_EMBED_BATCH_SIZE,
        )
        self._embed_batch_size = max(1, int(embed_batch_size))
        self._fetch_batch_size = self._embed_batch_size

        max_inflight = self.concurrency * MAX_INFLIGHT_MULTIPLIER
        self._inflight_semaphore = threading.BoundedSemaphore(max_inflight)
        self._inflight_lock = threading.Lock()
        self._inflight_conversations: set[UUID] = set()

    def enqueue_conversation(self, conversation_id: UUID) -> None:
        """
        Enqueue a conversation for embedding.
        """
        with self._inflight_lock:
            if conversation_id in self._inflight_conversations:
                logger.info(
                    "Conversation %s already queued for embedding", conversation_id
                )
                return
            if not self._inflight_semaphore.acquire(blocking=False):
                logger.warning(
                    "Embedding queue full; skipping conversation %s", conversation_id
                )
                return
            self._inflight_conversations.add(conversation_id)
        try:
            self.executor.submit(self._run_embedding_for_conversation, conversation_id)
        except Exception as exc:
            logger.warning(
                "Failed to enqueue embedding job for %s: %s", conversation_id, exc
            )
            self._release_conversation(conversation_id)

    def _release_conversation(self, conversation_id: UUID) -> None:
        with self._inflight_lock:
            if conversation_id in self._inflight_conversations:
                self._inflight_conversations.remove(conversation_id)
                self._inflight_semaphore.release()

    @staticmethod
    def _normalize_embedding(value: Any) -> list[float] | None:
        if value is None:
            return None
        if hasattr(value, "tolist"):
            value = value.tolist()
        if isinstance(value, (list, tuple)):
            try:
                return [float(item) for item in value]
            except (TypeError, ValueError):
                return None
        return None

    def _run_embedding_for_conversation(self, conversation_id: UUID) -> None:
        """
        Generate embeddings for all chunks in a conversation.
        """
        logger.info("Generating embeddings for conversation %s", conversation_id)
        session = None
        try:
            session = self.Session()
            result = session.execute(
                text(
                    "SELECT chunk_id, text FROM chunks WHERE conversation_id = :conversation_id AND embedding IS NULL"
                ),
                {"conversation_id": str(conversation_id)},
            )
            update_stmt = text(
                "UPDATE chunks SET embedding = :embedding WHERE chunk_id = :chunk_id"
            )

            total_updated = 0
            found_any = False

            while True:
                rows = result.fetchmany(self._fetch_batch_size)
                if not rows:
                    break
                found_any = True

                valid_pairs: list[tuple[Any, str]] = []
                skipped = 0
                for row in rows:
                    text_value = getattr(row, "text", None)
                    if text_value is None:
                        skipped += 1
                        continue
                    if isinstance(text_value, str):
                        if not text_value.strip():
                            skipped += 1
                            continue
                        normalized_text = text_value
                    else:
                        normalized_text = str(text_value)
                    valid_pairs.append((row.chunk_id, normalized_text))

                if skipped:
                    logger.debug(
                        "Skipping %d empty chunks for conversation %s",
                        skipped,
                        conversation_id,
                    )

                if not valid_pairs:
                    continue

                for start in range(0, len(valid_pairs), self._embed_batch_size):
                    batch = valid_pairs[start : start + self._embed_batch_size]
                    batch_ids = [pair[0] for pair in batch]
                    batch_texts = [pair[1] for pair in batch]

                    embeddings = self.embedding_client.embed_batch(batch_texts)
                    if len(embeddings) != len(batch_texts):
                        raise ValueError(
                            "Embedding count mismatch: expected "
                            f"{len(batch_texts)}, got {len(embeddings)}"
                        )

                    update_payload: list[dict[str, Any]] = []
                    for chunk_id, embedding in zip(batch_ids, embeddings, strict=True):
                        normalized_embedding = self._normalize_embedding(embedding)
                        if normalized_embedding is None:
                            logger.warning(
                                "Skipping chunk %s due to invalid embedding",
                                chunk_id,
                            )
                            continue
                        update_payload.append(
                            {
                                "embedding": normalized_embedding,
                                "chunk_id": chunk_id,
                            }
                        )

                    if update_payload:
                        session.execute(update_stmt, update_payload)
                        total_updated += len(update_payload)

            if not found_any:
                logger.info("No chunks to embed for conversation %s", conversation_id)
                return

            session.commit()
            logger.info(
                "Successfully embedded %d chunks for conversation %s",
                total_updated,
                conversation_id,
            )
        except Exception:
            if session is not None:
                session.rollback()
            logger.exception("Error embedding conversation %s", conversation_id)
            raise
        finally:
            if session is not None:
                session.close()
            self._release_conversation(conversation_id)

    def shutdown(self):
        """
        Shutdown the thread pool executor.
        """
        self.executor.shutdown(wait=True)
        try:
            self.engine.dispose()
        except Exception:
            logger.warning("Failed to dispose SQLAlchemy engine", exc_info=True)
