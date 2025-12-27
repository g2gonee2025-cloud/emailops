"""
Unified Indexer for Embeddings.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import List
from uuid import UUID

from cortex.config.loader import get_config
from cortex.embeddings.client import EmbeddingsClient
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

logger = logging.getLogger(__name__)


class Indexer:
    """
    Handles embedding and indexing of conversation chunks.
    """

    def __init__(self, concurrency: int = 4):
        self.concurrency = concurrency
        self.executor = ThreadPoolExecutor(max_workers=self.concurrency)
        self.embedding_client = EmbeddingsClient()

        config = get_config()
        db_url = config.database.url
        self.engine = create_engine(db_url)
        self.Session = sessionmaker(bind=self.engine)

    def enqueue_conversation(self, conversation_id: UUID) -> None:
        """
        Enqueue a conversation for embedding.
        """
        self.executor.submit(self._run_embedding_for_conversation, conversation_id)

    def _run_embedding_for_conversation(self, conversation_id: UUID) -> None:
        """
        Generate embeddings for all chunks in a conversation.
        """
        logger.info(f"Generating embeddings for conversation {conversation_id}")
        try:
            with self.Session() as session:
                rows = session.execute(
                    text(
                        "SELECT chunk_id, text FROM chunks WHERE conversation_id = :conversation_id AND embedding IS NULL"
                    ),
                    {"conversation_id": conversation_id},
                ).fetchall()

                if not rows:
                    logger.info(
                        f"No chunks to embed for conversation {conversation_id}"
                    )
                    return

                chunk_texts = [row.text for row in rows]
                chunk_ids = [row.chunk_id for row in rows]

                embeddings = self.embedding_client.embed_batch(chunk_texts)

                for chunk_id, embedding in zip(chunk_ids, embeddings):
                    session.execute(
                        text(
                            "UPDATE chunks SET embedding = :embedding WHERE chunk_id = :chunk_id"
                        ),
                        {"embedding": embedding, "chunk_id": chunk_id},
                    )
                session.commit()
                logger.info(
                    f"Successfully embedded {len(rows)} chunks for conversation {conversation_id}"
                )
        except Exception as e:
            logger.exception(f"Error embedding conversation {conversation_id}: {e}")

    def shutdown(self):
        """
        Shutdown the thread pool executor.
        """
        self.executor.shutdown(wait=True)
