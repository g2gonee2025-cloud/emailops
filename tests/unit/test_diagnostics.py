"""
Unit tests for diagnostics/diagnostics.py

Tests cover:
- Account validation and testing
- Index verification
- Consistency checking
- Field validation
"""

import json
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from diagnostics.diagnostics import (
    REQUIRED_FIELDS,
    VALID_DOCTYPES,
    check_index_consistency,
    test_account,
    verify_index_alignment,
)
from emailops.llm_runtime import VertexAccount


# ============================================================================
# test_account Tests
# ============================================================================

class TestAccountTesting:
    """Tests for test_account function."""
    
    def test_account_with_missing_credentials_returns_failure(self, sample_vertex_account):
        """Test that test_account fails when credentials file doesn't exist."""
        account = VertexAccount(
            project_id=sample_vertex_account["project_id"],
            credentials_path="/nonexistent/path/credentials.json",
            account_group=0
        )
        
        success, message = test_account(account)
        
        assert success is False
        assert "not found" in message.lower()
    
    @patch('diagnostics.diagnostics.reset_vertex_init')
    @patch('diagnostics.diagnostics._init_vertex')
    @patch('diagnostics.diagnostics.embed_texts')
    def test_account_with_valid_setup_returns_success(
        self, mock_embed, mock_init, mock_reset, mock_credentials_file, sample_vertex_account
    ):
        """Test that test_account succeeds with valid account and credentials."""
        # Setup mocks
        mock_reset.return_value = None
        mock_init.return_value = None
        # Return a non-empty numpy array with proper shape
        # The actual code checks: if embeddings and len(embeddings) > 0
        # Mock should return array with shape (n, 768) as expected
        mock_embeddings = np.random.randn(1, 768).astype(np.float32)
        mock_embed.return_value = mock_embeddings
        
        account = VertexAccount(
            project_id=sample_vertex_account["project_id"],
            credentials_path=str(mock_credentials_file),
            account_group=0
        )
        
        success, message = test_account(account)
        
        assert success is True
        assert message == "All tests passed"
        mock_init.assert_called_once()
        mock_embed.assert_called_once()
    
    @patch('diagnostics.diagnostics.reset_vertex_init')
    @patch('diagnostics.diagnostics._init_vertex')
    def test_account_with_init_failure_returns_error(
        self, mock_init, mock_reset, mock_credentials_file, sample_vertex_account
    ):
        """Test that test_account handles initialization failures."""
        mock_reset.return_value = None
        mock_init.side_effect = Exception("Init failed")
        
        account = VertexAccount(
            project_id=sample_vertex_account["project_id"],
            credentials_path=str(mock_credentials_file),
            account_group=0
        )
        
        success, message = test_account(account)
        
        assert success is False
        assert "Init failed" in message
    
    @patch('diagnostics.diagnostics.reset_vertex_init')
    @patch('diagnostics.diagnostics._init_vertex')
    @patch('diagnostics.diagnostics.embed_texts')
    def test_account_with_embedding_failure_returns_error(
        self, mock_embed, mock_init, mock_reset, mock_credentials_file, sample_vertex_account
    ):
        """Test that test_account handles embedding failures."""
        mock_reset.return_value = None
        mock_init.return_value = None
        mock_embed.side_effect = Exception("Embedding failed")
        
        account = VertexAccount(
            project_id=sample_vertex_account["project_id"],
            credentials_path=str(mock_credentials_file),
            account_group=0
        )
        
        success, message = test_account(account)
        
        assert success is False
        assert "Embedding failed" in message
    
    @patch('diagnostics.diagnostics.reset_vertex_init')
    @patch('diagnostics.diagnostics._init_vertex')
    @patch('diagnostics.diagnostics.embed_texts')
    def test_account_with_empty_embeddings_returns_failure(
        self, mock_embed, mock_init, mock_reset, mock_credentials_file, sample_vertex_account
    ):
        """Test that test_account fails when no embeddings are returned."""
        mock_reset.return_value = None
        mock_init.return_value = None
        # Return None to simulate no embeddings
        mock_embed.return_value = None
        
        account = VertexAccount(
            project_id=sample_vertex_account["project_id"],
            credentials_path=str(mock_credentials_file),
            account_group=0
        )
        
        success, message = test_account(account)
        
        assert success is False
        assert "No embeddings returned" in message


# ============================================================================
# verify_index_alignment Tests
# ============================================================================

class TestVerifyIndexAlignment:
    """Tests for verify_index_alignment function."""
    
    def test_verify_with_missing_index_dir_exits(self, temp_dir):
        """Test that verify_index_alignment exits when index directory missing."""
        with pytest.raises(SystemExit) as exc_info:
            verify_index_alignment(str(temp_dir))
        
        assert exc_info.value.code == 2
    
    def test_verify_with_missing_mapping_exits(self, mock_index_dir):
        """Test that verify_index_alignment exits when mapping.json missing."""
        # Create only index dir, no mapping file
        with pytest.raises(SystemExit) as exc_info:
            verify_index_alignment(str(mock_index_dir.parent))
        
        assert exc_info.value.code == 2
    
    def test_verify_with_valid_index_succeeds(self, mock_index_files):
        """Test that verify_index_alignment succeeds with valid index."""
        # Should complete without raising SystemExit
        verify_index_alignment(str(mock_index_files["index_dir"].parent))
        
        # If we get here, verification passed
        assert True
    
    def test_verify_with_mismatched_counts_exits(self, mock_index_dir, sample_mapping_data):
        """Test that verify_index_alignment exits when counts don't match."""
        # Create mapping with 2 entries
        mapping_path = mock_index_dir / "mapping.json"
        with open(mapping_path, "w") as f:
            json.dump(sample_mapping_data, f)
        
        # Create embeddings with 3 entries (mismatch)
        embeddings_path = mock_index_dir / "embeddings.npy"
        embeddings = np.random.randn(3, 768).astype(np.float32)
        np.save(embeddings_path, embeddings)
        
        with pytest.raises(SystemExit) as exc_info:
            verify_index_alignment(str(mock_index_dir.parent))
        
        assert exc_info.value.code == 2
    
    def test_verify_with_missing_required_field_exits(self, mock_index_dir, sample_embeddings):
        """Test that verify_index_alignment exits when required field missing."""
        # Create mapping with missing field
        incomplete_mapping = [
            {
                "id": "test_1",
                "path": "/test/path",
                # Missing other required fields
            }
        ]
        
        mapping_path = mock_index_dir / "mapping.json"
        with open(mapping_path, "w") as f:
            json.dump(incomplete_mapping, f)
        
        embeddings_path = mock_index_dir / "embeddings.npy"
        embeddings = sample_embeddings[:1]  # Match count
        np.save(embeddings_path, embeddings)
        
        with pytest.raises(SystemExit) as exc_info:
            verify_index_alignment(str(mock_index_dir.parent))
        
        assert exc_info.value.code == 2
    
    def test_verify_with_duplicate_ids_exits(self, mock_index_dir, sample_embeddings):
        """Test that verify_index_alignment exits when duplicate IDs found."""
        # Create mapping with duplicate IDs
        duplicate_mapping = [
            {
                "id": "duplicate_id",
                "path": "/test/path1",
                "conv_id": "conv_1",
                "doc_type": "conversation",
                "subject": "Test",
                "snippet": "Test snippet",
                "modified_time": "2024-01-01T12:00:00Z"
            },
            {
                "id": "duplicate_id",  # Duplicate
                "path": "/test/path2",
                "conv_id": "conv_2",
                "doc_type": "conversation",
                "subject": "Test 2",
                "snippet": "Test snippet 2",
                "modified_time": "2024-01-01T12:00:00Z"
            }
        ]
        
        mapping_path = mock_index_dir / "mapping.json"
        with open(mapping_path, "w") as f:
            json.dump(duplicate_mapping, f)
        
        embeddings_path = mock_index_dir / "embeddings.npy"
        np.save(embeddings_path, sample_embeddings)
        
        with pytest.raises(SystemExit) as exc_info:
            verify_index_alignment(str(mock_index_dir.parent))
        
        assert exc_info.value.code == 2
    
    def test_verify_with_invalid_doctype_exits(self, mock_index_dir, sample_embeddings):
        """Test that verify_index_alignment exits with invalid doc_type."""
        invalid_mapping = [
            {
                "id": "test_1",
                "path": "/test/path",
                "conv_id": "conv_1",
                "doc_type": "invalid_type",  # Invalid
                "subject": "Test",
                "snippet": "Test snippet",
                "modified_time": "2024-01-01T12:00:00Z"
            }
        ]
        
        mapping_path = mock_index_dir / "mapping.json"
        with open(mapping_path, "w") as f:
            json.dump(invalid_mapping, f)
        
        embeddings_path = mock_index_dir / "embeddings.npy"
        embeddings = sample_embeddings[:1]
        np.save(embeddings_path, embeddings)
        
        with pytest.raises(SystemExit) as exc_info:
            verify_index_alignment(str(mock_index_dir.parent))
        
        assert exc_info.value.code == 2


# ============================================================================
# check_index_consistency Tests
# ============================================================================

class TestCheckIndexConsistency:
    """Tests for check_index_consistency function."""
    
    def test_check_with_missing_index_returns_error(self, temp_dir):
        """Test that check_index_consistency reports missing index."""
        result = check_index_consistency(temp_dir)
        
        assert "errors" in result
        assert len(result["errors"]) > 0
        assert result["checks"]["index_exists"] is False
    
    def test_check_with_valid_index_returns_healthy(self, mock_index_files):
        """Test that check_index_consistency reports healthy for valid index."""
        result = check_index_consistency(mock_index_files["index_dir"].parent)
        
        assert result["status"] == "HEALTHY"
        assert result["checks"]["index_exists"] is True
        assert result["checks"]["mapping_exists"] is True
        assert result["checks"]["embeddings_exists"] is True
        assert result["checks"]["counts_aligned"] is True
    
    def test_check_with_missing_mapping_returns_errors(self, mock_index_dir):
        """Test that check_index_consistency detects missing mapping."""
        # Create only embeddings, no mapping
        embeddings_path = mock_index_dir / "embeddings.npy"
        np.save(embeddings_path, np.random.randn(5, 768).astype(np.float32))
        
        result = check_index_consistency(mock_index_dir.parent)
        
        assert result["checks"]["mapping_exists"] is False
        assert "mapping.json not found" in result["errors"]
    
    def test_check_with_mismatched_counts_returns_critical(
        self, mock_index_dir, sample_mapping_data
    ):
        """Test that check_index_consistency detects count mismatch."""
        # Create mapping with 2 entries
        mapping_path = mock_index_dir / "mapping.json"
        with open(mapping_path, "w") as f:
            json.dump(sample_mapping_data, f)
        
        # Create embeddings with 5 entries (mismatch)
        embeddings_path = mock_index_dir / "embeddings.npy"
        np.save(embeddings_path, np.random.randn(5, 768).astype(np.float32))
        
        result = check_index_consistency(mock_index_dir.parent)
        
        assert result["status"] == "CRITICAL"
        assert result["checks"]["counts_aligned"] is False
        assert any("mismatch" in err.lower() for err in result["errors"])
    
    def test_check_with_duplicate_ids_returns_error(
        self, mock_index_dir, sample_embeddings
    ):
        """Test that check_index_consistency detects duplicate IDs."""
        duplicate_mapping = [
            {"id": "dup", "path": "/a", "conv_id": "c1", "doc_type": "conversation",
             "subject": "S1", "snippet": "Sn1", "modified_time": "2024-01-01T12:00:00Z"},
            {"id": "dup", "path": "/b", "conv_id": "c2", "doc_type": "conversation",
             "subject": "S2", "snippet": "Sn2", "modified_time": "2024-01-01T12:00:00Z"}
        ]
        
        mapping_path = mock_index_dir / "mapping.json"
        with open(mapping_path, "w") as f:
            json.dump(duplicate_mapping, f)
        
        embeddings_path = mock_index_dir / "embeddings.npy"
        np.save(embeddings_path, sample_embeddings)
        
        result = check_index_consistency(mock_index_dir.parent)
        
        assert result["checks"]["ids_unique"] is False
        assert any("duplicate" in err.lower() for err in result["errors"])
    
    def test_check_includes_timestamp(self, mock_index_files):
        """Test that check_index_consistency includes timestamp."""
        result = check_index_consistency(mock_index_files["index_dir"].parent)
        
        assert "timestamp" in result
        # Should be parseable as datetime
        datetime.fromisoformat(result["timestamp"])
    
    def test_check_includes_recommendations_for_errors(self, mock_index_dir):
        """Test that check_index_consistency provides recommendations."""
        # Create scenario with missing files
        result = check_index_consistency(mock_index_dir.parent)
        
        assert "recommendations" in result
        if result["errors"]:
            assert len(result["recommendations"]) > 0
    
    def test_check_validates_meta_dimensions(self, mock_index_files):
        """Test that check_index_consistency validates meta.json dimensions."""
        # Modify meta.json to have wrong dimensions
        meta_path = mock_index_files["meta"]
        with open(meta_path, "r") as f:
            meta = json.load(f)
        
        meta["actual_dimensions"] = 512  # Wrong dimension
        
        with open(meta_path, "w") as f:
            json.dump(meta, f)
        
        result = check_index_consistency(mock_index_files["index_dir"].parent)
        
        assert result["checks"]["meta_dimensions_match"] is False
        assert any("dimension" in warn.lower() for warn in result.get("warnings", []))


# ============================================================================
# Field Validation Tests
# ============================================================================

class TestFieldValidation:
    """Tests for field validation constants and logic."""
    
    def test_required_fields_constant_is_complete(self):
        """Test that REQUIRED_FIELDS contains all necessary fields."""
        expected_fields = [
            "id", "path", "conv_id", "doc_type",
            "subject", "snippet", "modified_time"
        ]
        
        assert all(field in REQUIRED_FIELDS for field in expected_fields)
    
    def test_valid_doctypes_constant_is_correct(self):
        """Test that VALID_DOCTYPES contains expected values."""
        assert "conversation" in VALID_DOCTYPES
        assert "attachment" in VALID_DOCTYPES
        assert len(VALID_DOCTYPES) == 2
    
    def test_mapping_entry_with_all_fields_is_valid(self, sample_mapping_data):
        """Test that sample mapping data has all required fields."""
        for entry in sample_mapping_data:
            for field in REQUIRED_FIELDS:
                assert field in entry, f"Missing field: {field}"
    
    def test_mapping_entry_doctype_validation(self):
        """Test doc_type validation logic."""
        valid_types = ["conversation", "attachment"]
        
        for doc_type in valid_types:
            assert doc_type in VALID_DOCTYPES
        
        invalid_types = ["email", "document", "message"]
        for doc_type in invalid_types:
            assert doc_type not in VALID_DOCTYPES


# ============================================================================
# Edge Case Tests
# ============================================================================

class TestDiagnosticsEdgeCases:
    """Edge case tests for diagnostics functions."""
    
    def test_check_with_empty_mapping_returns_healthy(self, mock_index_dir):
        """Test that check_index_consistency handles empty mapping."""
        # Create empty mapping and embeddings
        mapping_path = mock_index_dir / "mapping.json"
        with open(mapping_path, "w") as f:
            json.dump([], f)
        
        embeddings_path = mock_index_dir / "embeddings.npy"
        np.save(embeddings_path, np.zeros((0, 768), dtype=np.float32))
        
        result = check_index_consistency(mock_index_dir.parent)
        
        assert result["checks"]["mapping_valid"] is True
        assert result["checks"]["mapping_count"] == 0
        assert result["checks"]["counts_aligned"] is True
    
    def test_check_with_corrupted_json_returns_error(self, mock_index_dir):
        """Test that check_index_consistency handles corrupted JSON."""
        # Create corrupted mapping file
        mapping_path = mock_index_dir / "mapping.json"
        mapping_path.write_text("{ invalid json }")
        
        embeddings_path = mock_index_dir / "embeddings.npy"
        np.save(embeddings_path, np.random.randn(5, 768).astype(np.float32))
        
        result = check_index_consistency(mock_index_dir.parent)
        
        assert result["checks"]["mapping_valid"] is False
        assert any("failed to load mapping" in err.lower() for err in result["errors"])
    
    def test_verify_with_non_string_snippet_exits(self, mock_index_dir, sample_embeddings):
        """Test that verify_index_alignment catches non-string snippet."""
        invalid_mapping = [
            {
                "id": "test_1",
                "path": "/test/path",
                "conv_id": "conv_1",
                "doc_type": "conversation",
                "subject": "Test",
                "snippet": 12345,  # Not a string
                "modified_time": "2024-01-01T12:00:00Z"
            }
        ]
        
        mapping_path = mock_index_dir / "mapping.json"
        with open(mapping_path, "w") as f:
            json.dump(invalid_mapping, f)
        
        embeddings_path = mock_index_dir / "embeddings.npy"
        embeddings = sample_embeddings[:1]
        np.save(embeddings_path, embeddings)
        
        with pytest.raises(SystemExit) as exc_info:
            verify_index_alignment(str(mock_index_dir.parent))
        
        assert exc_info.value.code == 2