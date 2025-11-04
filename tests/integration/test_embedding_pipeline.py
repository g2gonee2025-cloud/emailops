"""
Integration tests for embedding generation pipeline

Tests cover:
- Embedding generation workflow
"""

import os
from unittest.mock import patch

import numpy as np
import pytest

from emailops.llm_client_shim import embed_texts

# ============================================================================
# Embedding Generation Workflow Tests
# ============================================================================


@pytest.mark.integration
@pytest.mark.requires_gcp
class TestEmbeddingGeneration:
    """Integration tests for embedding generation."""

    @patch('emailops.llm_runtime._init_vertex')
    @patch('emailops.llm_runtime._embed_vertex')
    def test_embed_texts_with_vertex_provider(self, mock_embed_vertex, _mock_init_vertex):
        """
        Test basic embedding generation with vertex provider.

        Verifies:
        1. Embeddings are generated for text inputs
        2. Output shape is correct
        3. Embeddings are normalized
        """
        # Mock the vertex embedding to return normalized vectors
        mock_embeddings = np.random.randn(3, 3072).astype(np.float32)
        norms = np.linalg.norm(mock_embeddings, axis=1, keepdims=True)
        mock_embeddings = mock_embeddings / norms
        mock_embed_vertex.return_value = mock_embeddings

        # Set up minimal environment
        os.environ['GCP_PROJECT'] = 'test-project'
        os.environ['EMBED_PROVIDER'] = 'vertex'

        texts = ["Hello world", "Test embedding", "Another text"]

        embeddings = embed_texts(texts)

        assert embeddings.shape[0] == 3
        assert embeddings.shape[1] == 3072

        # Check normalization
        norms = np.linalg.norm(embeddings, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-5)
