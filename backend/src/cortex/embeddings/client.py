"""
Embeddings Client.

Provides a stable interface for embedding operations.
"""
from __future__ import annotations

from typing import List

from cortex.llm.client import embed_texts as _embed_texts


class EmbeddingsClient:
    """
    Client for generating text embeddings.
    """

    def embed(self, text: str) -> List[float]:
        """Embed a single text string."""
        return self.embed_texts([text])[0]

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of text strings (alias for embed_texts)."""
        return self.embed_texts(texts)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of texts and return python lists."""
        if not texts:
            return []
        result = _embed_texts(texts)
        return result.tolist() if hasattr(result, "tolist") else result

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
