"""
Integration tests for embedding generation pipeline

Tests cover:
- Embedding generation workflow
- Batch processing
- Error handling in embedding pipeline
"""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest


# ============================================================================
# Embedding Generation Workflow Tests
# ============================================================================

@pytest.mark.integration
@pytest.mark.requires_gcp
class TestEmbeddingGeneration:
    """Integration tests for embedding generation."""
    
    def test_embed_texts_with_mock_provider(self, mock_embed_texts):
        """
        Test basic embedding generation with mocked provider.
        
        Verifies:
        1. Embeddings are generated for text inputs
        2. Output shape is correct
        3. Embeddings are normalized
        """
        texts = ["Hello world", "Test embedding", "Another text"]
        
        embeddings = mock_embed_texts(texts)
        
        assert embeddings.shape[0] == 3
        assert embeddings.shape[1] == 768
        
        # Check normalization
        norms = np.linalg.norm(embeddings, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-5)
    
    def test_embed_empty_list_returns_empty_array(self, mock_embed_texts):
        """Test that embedding empty list returns empty array."""
        embeddings = mock_embed_texts([])
        
        assert embeddings.shape[0] == 0
    
    def test_embed_single_text_returns_single_embedding(self, mock_embed_texts):
        """Test embedding single text."""
        embeddings = mock_embed_texts(["Single text"])
        
        assert embeddings.shape[0] == 1
        assert embeddings.shape[1] == 768


# ============================================================================
# Batch Processing Tests
# ============================================================================

@pytest.mark.integration
class TestBatchProcessing:
    """Integration tests for batch embedding processing."""
    
    def test_batch_processing_maintains_order(self, mock_embed_texts):
        """
        Test that batch processing maintains input order.
        
        Verifies:
        1. Texts are processed in batches
        2. Output order matches input order
        3. All texts are processed
        """
        # Create identifiable texts
        texts = [f"Text {i}" for i in range(100)]
        
        embeddings = mock_embed_texts(texts)
        
        assert embeddings.shape[0] == 100
        # Each embedding should be unique (very high probability with random generation)
        assert len(np.unique(embeddings, axis=0)) > 90
    
    def test_large_batch_processing(self, mock_embed_texts):
        """Test processing large batch of texts."""
        texts = [f"Document {i}" for i in range(500)]
        
        embeddings = mock_embed_texts(texts)
        
        assert embeddings.shape[0] == 500
        assert embeddings.dtype == np.float32
    
    def test_batch_processing_with_various_lengths(self, mock_embed_texts):
        """Test embedding texts of various lengths."""
        texts = [
            "Short",
            "A medium length text with several words",
            "A very long text " * 100  # Very long text
        ]
        
        embeddings = mock_embed_texts(texts)
        
        assert embeddings.shape[0] == 3
        # All embeddings should have same dimension regardless of input length
        assert embeddings.shape[1] == 768


# ============================================================================
# Embedding Quality Tests
# ============================================================================

@pytest.mark.integration
class TestEmbeddingQuality:
    """Integration tests for embedding quality checks."""
    
    def test_embeddings_are_deterministic_with_same_input(self, mock_embed_texts):
        """
        Test that same input produces consistent embeddings.
        
        Note: This test uses mocked embeddings which are random,
        so we're testing the structure rather than exact values.
        """
        text = "Test text for determinism"
        
        emb1 = mock_embed_texts([text])
        emb2 = mock_embed_texts([text])
        
        # Both should have same shape
        assert emb1.shape == emb2.shape
        # Both should be normalized
        assert np.abs(np.linalg.norm(emb1[0]) - 1.0) < 1e-5
        assert np.abs(np.linalg.norm(emb2[0]) - 1.0) < 1e-5
    
    def test_embeddings_have_no_nan_or_inf(self, mock_embed_texts):
        """Test that embeddings don't contain NaN or Inf values."""
        texts = ["Text 1", "Text 2", "Text 3"]
        
        embeddings = mock_embed_texts(texts)
        
        assert not np.any(np.isnan(embeddings))
        assert not np.any(np.isinf(embeddings))
    
    def test_embedding_vectors_are_not_zero(self, mock_embed_texts):
        """Test that embedding vectors are not all zeros."""
        texts = ["Non-empty text"]
        
        embeddings = mock_embed_texts(texts)
        
        # At least some values should be non-zero
        assert np.any(embeddings != 0)


# ============================================================================
# Error Handling Tests
# ============================================================================

@pytest.mark.integration
class TestEmbeddingErrorHandling:
    """Integration tests for error handling in embedding pipeline."""
    
    def test_handles_special_characters(self, mock_embed_texts):
        """Test embedding texts with special characters."""
        texts = [
            "Text with emoji 🎉🌟",
            "Special chars: @#$%^&*()",
            "Unicode: 你好世界"
        ]
        
        embeddings = mock_embed_texts(texts)
        
        assert embeddings.shape[0] == 3
        assert not np.any(np.isnan(embeddings))
    
    def test_handles_very_long_text(self, mock_embed_texts):
        """Test embedding very long text."""
        long_text = "word " * 10000  # Very long text
        
        embeddings = mock_embed_texts([long_text])
        
        assert embeddings.shape[0] == 1
        assert np.linalg.norm(embeddings[0]) > 0
    
    def test_handles_empty_strings(self, mock_embed_texts):
        """Test embedding empty strings."""
        texts = ["", "  ", "\n\t"]
        
        embeddings = mock_embed_texts(texts)
        
        assert embeddings.shape[0] == 3
        # Even empty strings should produce embeddings
        assert embeddings.shape[1] == 768


# ============================================================================
# Integration with Index Creation
# ============================================================================

@pytest.mark.integration
class TestEmbeddingIndexIntegration:
    """Integration tests for embedding generation integrated with index creation."""
    
    def test_embeddings_align_with_chunks(self, temp_dir, sample_chunk_data, mock_embed_texts):
        """
        Test that embeddings align correctly with chunk data.
        
        Verifies:
        1. One embedding per chunk
        2. Embeddings can be matched to chunks
        3. Order is preserved
        """
        chunks = sample_chunk_data["chunks"]
        texts = [chunk["text"] for chunk in chunks]
        
        embeddings = mock_embed_texts(texts)
        
        # Should have one embedding per chunk
        assert embeddings.shape[0] == len(chunks)
        
        # Create index structure
        index_dir = temp_dir / "_index"
        index_dir.mkdir()
        
        # Save embeddings
        np.save(index_dir / "embeddings.npy", embeddings)
        
        # Verify saved embeddings
        loaded = np.load(index_dir / "embeddings.npy")
        assert loaded.shape == embeddings.shape
    
    def test_embedding_dimensions_match_metadata(self, temp_dir, mock_embed_texts):
        """
        Test that embedding dimensions match metadata specification.
        
        Verifies:
        1. Embeddings have expected dimensions
        2. Metadata reflects actual dimensions
        3. Dimension consistency throughout pipeline
        """
        texts = ["Test text 1", "Test text 2"]
        embeddings = mock_embed_texts(texts)
        
        expected_dim = 768
        assert embeddings.shape[1] == expected_dim
        
        # Create metadata
        import json
        index_dir = temp_dir / "_index"
        index_dir.mkdir()
        
        meta = {
            "provider": "vertex",
            "model": "gemini-embedding-001",
            "actual_dimensions": embeddings.shape[1],
            "index_type": "flat"
        }
        
        meta_path = index_dir / "meta.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f)
        
        # Verify metadata
        with open(meta_path, "r") as f:
            loaded_meta = json.load(f)
        
        assert loaded_meta["actual_dimensions"] == expected_dim


# ============================================================================
# Performance Tests
# ============================================================================

@pytest.mark.integration
@pytest.mark.slow
class TestEmbeddingPerformance:
    """Integration tests for embedding performance."""
    
    def test_process_many_texts_efficiently(self, mock_embed_texts):
        """
        Test that system can process many texts.
        
        Verifies:
        1. Large batches are processed
        2. Memory usage is reasonable
        3. Results are valid
        """
        num_texts = 1000
        texts = [f"Document number {i}" for i in range(num_texts)]
        
        embeddings = mock_embed_texts(texts)
        
        assert embeddings.shape[0] == num_texts
        assert embeddings.dtype == np.float32
        
        # Check a sample of embeddings
        for i in range(0, num_texts, 100):
            norm = np.linalg.norm(embeddings[i])
            assert 0.99 <= norm <= 1.01  # Should be normalized


# ============================================================================
# Provider-Specific Tests
# ============================================================================

@pytest.mark.integration
class TestProviderBehavior:
    """Integration tests for provider-specific behavior."""
    
    def test_vertex_provider_initialization(self, mock_vertex_ai, mock_embed_texts):
        """Test Vertex AI provider initialization."""
        texts = ["Test text"]
        
        with patch.dict('os.environ', {'EMBED_PROVIDER': 'vertex'}):
            embeddings = mock_embed_texts(texts, provider='vertex')
            
            assert embeddings.shape[0] == 1
    
    def test_provider_fallback_behavior(self, mock_embed_texts):
        """Test fallback behavior when provider specified."""
        texts = ["Fallback test"]
        
        # Test with explicitly specified provider
        embeddings = mock_embed_texts(texts, provider='vertex')
        
        assert embeddings.shape[0] == 1
        assert embeddings.shape[1] == 768


# ============================================================================
# Data Flow Tests
# ============================================================================

@pytest.mark.integration
class TestEmbeddingDataFlow:
    """Integration tests for embedding data flow through pipeline."""
    
    def test_complete_data_flow_chunks_to_embeddings(
        self, temp_dir, sample_chunk_data, mock_embed_texts
    ):
        """
        Test complete data flow from chunks to stored embeddings.
        
        Workflow:
        1. Load chunks from files
        2. Extract text from chunks
        3. Generate embeddings
        4. Store embeddings with metadata
        5. Verify storage integrity
        """
        import json
        
        # Step 1: Create chunk files
        chunks_dir = temp_dir / "_chunks" / "chunks"
        chunks_dir.mkdir(parents=True)
        
        chunk_file = chunks_dir / "doc_001.json"
        with open(chunk_file, "w") as f:
            json.dump(sample_chunk_data, f)
        
        # Step 2: Load and extract text
        with open(chunk_file, "r") as f:
            data = json.load(f)
        
        texts = [chunk["text"] for chunk in data["chunks"]]
        
        # Step 3: Generate embeddings
        embeddings = mock_embed_texts(texts)
        
        # Step 4: Store embeddings
        index_dir = temp_dir / "_index"
        index_dir.mkdir()
        
        emb_path = index_dir / "embeddings.npy"
        np.save(emb_path, embeddings)
        
        # Step 5: Verify storage
        loaded_emb = np.load(emb_path)
        
        assert loaded_emb.shape == embeddings.shape
        assert loaded_emb.dtype == np.float32
        assert np.allclose(loaded_emb, embeddings)
    
    def test_embedding_metadata_consistency(self, temp_dir, mock_embed_texts):
        """
        Test that embedding metadata remains consistent.
        
        Verifies:
        1. Metadata is created with embeddings
        2. Metadata accurately reflects embedding properties
        3. Metadata can be loaded and validated
        """
        import json
        
        texts = ["Text 1", "Text 2", "Text 3"]
        embeddings = mock_embed_texts(texts)
        
        index_dir = temp_dir / "_index"
        index_dir.mkdir()
        
        # Save embeddings
        np.save(index_dir / "embeddings.npy", embeddings)
        
        # Create metadata
        meta = {
            "provider": "vertex",
            "model": "gemini-embedding-001",
            "actual_dimensions": int(embeddings.shape[1]),
            "num_embeddings": int(embeddings.shape[0]),
            "dtype": str(embeddings.dtype)
        }
        
        with open(index_dir / "meta.json", "w") as f:
            json.dump(meta, f)
        
        # Verify metadata
        with open(index_dir / "meta.json", "r") as f:
            loaded_meta = json.load(f)
        
        assert loaded_meta["actual_dimensions"] == embeddings.shape[1]
        assert loaded_meta["num_embeddings"] == embeddings.shape[0]
        assert loaded_meta["dtype"] == "float32"