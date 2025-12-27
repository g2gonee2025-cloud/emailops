"""
Embeddings Client.

Provides a stable interface for embedding operations.
"""

from __future__ import annotations

from typing import Final, List

from cortex.llm.client import embed_texts as _embed_texts

DEFAULT_EMBED_BATCH_SIZE: Final[int] = 50
"""
Default batch size for embedding operations."""


class EmbeddingsClient:
    """
    Client for generating text embeddings.
    """

    def embed(self, text: str) -> List[float]:
        """Embed a single text string."""
        result = self.embed_texts([text])
        if not result:
            return []  # P2 Fix: Handle empty result
        return result[0]

    BATCH_SIZE = 50

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of text strings (alias for embed_texts)."""
        results = []
        for i in range(0, len(texts), self.BATCH_SIZE):
            batch = texts[i : i + self.BATCH_SIZE]
            results.extend(self.embed_texts(batch))
        return results

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of texts and return python lists."""
        if not texts:
            return []
        result = _embed_texts(texts)
        if result is None:  # P2 Fix: Handle None return
            return []
        return result.tolist() if hasattr(result, "tolist") else list(result)

    @staticmethod
    def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
        """Compute cosine similarity between two python lists."""
        if not vec_a or not vec_b or len(vec_a) != len(vec_b):
            return 0.0
        dot = sum(a * b for a, b in zip(vec_a, vec_b))
        mag_a = sum(a * a for a in vec_a) ** 0.5
        mag_b = sum(b * b for b in vec_b) ** 0.5
        if mag_a == 0 or mag_b == 0:
            return 0.0
        return dot / (mag_a * mag_b)


def get_embedding(text: str) -> List[float]:
    """
    Helper function to get embedding for a single text.
    """
    client = EmbeddingsClient()
    return client.embed(text)
