"""
Embeddings Client.

Provides a stable interface for embedding operations.
"""
from __future__ import annotations

from typing import List

from cortex.llm import runtime

class EmbeddingsClient:
    """
    Client for generating text embeddings.
    """
    
    def embed(self, text: str) -> List[float]:
        """
        Embed a single text string.
        """
        result = runtime.embed_texts([text])
        return result[0].tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a batch of text strings.
        """
        result = runtime.embed_texts(texts)
        return result.tolist()

def get_embedding(text: str) -> List[float]:
    """
    Helper function to get embedding for a single text.
    """
    client = EmbeddingsClient()
    return client.embed(text)
