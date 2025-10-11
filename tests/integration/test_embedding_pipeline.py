"""
Integration tests for embedding generation pipeline

Tests cover:
- Embedding generation workflow
"""

import numpy as np
import pytest

from emailops.llm_client import embed_texts

# ============================================================================
# Embedding Generation Workflow Tests
# ============================================================================

@pytest.mark.integration
@pytest.mark.requires_gcp
class TestEmbeddingGeneration:
    """Integration tests for embedding generation."""

    def test_embed_texts_with_vertex_provider(self):
        """
        Test basic embedding generation with vertex provider.

        Verifies:
        1. Embeddings are generated for text inputs
        2. Output shape is correct
        3. Embeddings are normalized
        """
        texts = ["Hello world", "Test embedding", "Another text"]

        embeddings = embed_texts(texts)

        assert embeddings.shape[0] == 3
        assert embeddings.shape[1] == 768

        # Check normalization
        norms = np.linalg.norm(embeddings, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-5)
