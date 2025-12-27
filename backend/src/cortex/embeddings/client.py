"""
Embeddings Client.

Provides a stable interface for embedding operations.
"""

from __future__ import annotations

import threading
from typing import Optional, cast

import numpy as np
from cortex.llm.client import embed_texts as _embed_texts


class EmbeddingsClient:
    """
    Client for generating text embeddings.

    NOTE: This is a singleton. Use get_embeddings_client() to get the instance.
    """

    def embed(self, text: str) -> list[float]:
        """Embed a single text string."""
        if not text or not text.strip():
            return []

        result = self.embed_texts([text])
        return result[0] if result else []

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a list of texts and return python lists.
        Delegates to the LLM runtime, which handles batching and resilience.
        Maintains the same number of elements in the output list as the input list.
        """
        if not texts:
            return []

        # Identify non-empty texts to be embedded while keeping their original indices
        indices_and_texts_to_embed = [
            (i, t) for i, t in enumerate(texts) if t and t.strip()
        ]

        if not indices_and_texts_to_embed:
            # All texts are empty or whitespace
            return [[] for _ in texts]

        indices, texts_to_embed = zip(*indices_and_texts_to_embed)

        # Note: LLM runtime handles resilience (retry, circuit breaker)
        embeddings_array = _embed_texts(list(texts_to_embed))

        # Initialize a result list with placeholders for all original texts
        results: list[list[float]] = [[] for _ in texts]

        # If embedding fails or returns unexpected shape, return list of empty lists
        if (
            embeddings_array is None
            or embeddings_array.size == 0
            or len(embeddings_array) != len(texts_to_embed)
        ):
            return results

        embedded_vectors = cast(list[list[float]], embeddings_array.tolist())

        # Place the generated embeddings back into the correct positions
        for i, embedding in zip(indices, embedded_vectors):
            results[i] = embedding

        return results

    @staticmethod
    def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
        """Compute cosine similarity between two python lists."""
        if not vec_a or not vec_b or len(vec_a) != len(vec_b):
            return 0.0

        # For performance, convert to numpy arrays
        vec_a_np = np.asarray(vec_a, dtype=np.float32)
        vec_b_np = np.asarray(vec_b, dtype=np.float32)

        dot = np.dot(vec_a_np, vec_b_np)
        mag_a = np.linalg.norm(vec_a_np)
        mag_b = np.linalg.norm(vec_b_np)

        if mag_a == 0 or mag_b == 0:
            return 0.0

        return float(dot / (mag_a * mag_b))


# =============================================================================
# Singleton instance and accessor
# =============================================================================
_embeddings_client: EmbeddingsClient | None = None
_client_lock = threading.Lock()


def get_embeddings_client() -> EmbeddingsClient:
    """
    Get the singleton instance of the EmbeddingsClient.

    This function is thread-safe.
    """
    global _embeddings_client
    if _embeddings_client is None:
        with _client_lock:
            if _embeddings_client is None:
                _embeddings_client = EmbeddingsClient()
    return _embeddings_client


def get_embedding(text: str) -> list[float]:
    """
    Helper function to get embedding for a single text.
    """
    client = get_embeddings_client()
    return client.embed(text)
