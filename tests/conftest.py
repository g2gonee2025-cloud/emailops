"""
Shared pytest fixtures and configuration for EmailOps testing.

This module provides:
- Mock factories for common objects
- Test data generators
- Shared fixtures for file operations
- Mock GCP/Vertex AI clients
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

# ============================================================================
# Fixture Scopes and Cleanup
# ============================================================================


@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """
    Provides path to test data directory.

    Returns:
        Path to tests/test_data directory
    """
    data_dir = Path(__file__).parent / "test_data"
    data_dir.mkdir(exist_ok=True)
    return data_dir


@pytest.fixture
def temp_dir(tmp_path):
    """
    Provides a temporary directory for test operations.

    Yields:
        Path to temporary directory
    """
    yield tmp_path
    # Cleanup happens automatically via tmp_path


@pytest.fixture
def mock_index_dir(temp_dir):
    """
    Creates a mock index directory structure.

    Returns:
        Path to mock _index directory
    """
    index_dir = temp_dir / "_index"
    index_dir.mkdir(parents=True, exist_ok=True)
    return index_dir


# ============================================================================
# Mock Data Factories
# ============================================================================


@pytest.fixture
def sample_vertex_account():
    """
    Creates a sample VertexAccount object for testing.

    Returns:
        Dictionary with account configuration
    """
    return {
        "project_id": "test-project-123",
        "credentials_path": "secrets/test-credentials.json",
        "account_group": 0,
        "is_valid": True,
    }


@pytest.fixture
def sample_accounts_list(sample_vertex_account):
    """
    Creates a list of sample accounts.

    Returns:
        List of account dictionaries
    """
    return [
        sample_vertex_account,
        {
            "project_id": "test-project-456",
            "credentials_path": "secrets/test-credentials-2.json",
            "account_group": 1,
            "is_valid": True,
        },
    ]


@pytest.fixture
def sample_mapping_data():
    """
    Creates sample mapping.json data structure.

    Returns:
        List of mapping dictionaries
    """
    return [
        {
            "id": "conv_001::chunk_0",
            "path": "/path/to/conversation_1/Conversation.txt",
            "conv_id": "conv_001",
            "doc_type": "conversation",
            "subject": "Test Email Subject",
            "snippet": "This is a test email snippet",
            "modified_time": "2024-01-01T12:00:00Z",
            "chunk_index": 0,
        },
        {
            "id": "conv_001::attachment_1",
            "path": "/path/to/conversation_1/Attachments/doc.pdf",
            "conv_id": "conv_001",
            "doc_type": "attachment",
            "subject": "Test Email Subject",
            "snippet": "PDF attachment content",
            "modified_time": "2024-01-01T12:00:00Z",
        },
    ]


@pytest.fixture
def sample_embeddings():
    """
    Creates sample embeddings array.

    Returns:
        NumPy array of shape (2, 768) with normalized embeddings
    """
    embeddings = np.random.randn(2, 768).astype(np.float32)
    # Normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / norms


@pytest.fixture
def sample_chunk_data():
    """
    Creates sample chunk data structure.

    Returns:
        Dictionary with chunk data
    """
    return {
        "doc_id": "conversation_1/Conversation.txt",
        "num_chunks": 3,
        "chunks": [
            {
                "text": "This is the first chunk of text.",
                "chunk_index": 0,
                "start_char": 0,
                "end_char": 32,
            },
            {
                "text": "This is the second chunk of text.",
                "chunk_index": 1,
                "start_char": 32,
                "end_char": 65,
            },
            {
                "text": "This is the third chunk of text.",
                "chunk_index": 2,
                "start_char": 65,
                "end_char": 97,
            },
        ],
        "metadata": {"chunked_at": "2024-01-01T12:00:00", "original_size": 97},
    }


# ============================================================================
# File System Mocks
# ============================================================================


@pytest.fixture
def mock_index_files(mock_index_dir, sample_mapping_data, sample_embeddings):
    """
    Creates mock index files in the temp directory.

    Args:
        mock_index_dir: Path to mock index directory
        sample_mapping_data: Sample mapping data
        sample_embeddings: Sample embeddings array

    Returns:
        Dictionary with paths to created files
    """
    # Create mapping.json
    mapping_path = mock_index_dir / "mapping.json"
    with mapping_path.open("w", encoding="utf-8") as f:
        json.dump(sample_mapping_data, f, indent=2)

    # Create embeddings.npy
    embeddings_path = mock_index_dir / "embeddings.npy"
    np.save(embeddings_path, sample_embeddings)

    # Create meta.json
    meta_path = mock_index_dir / "meta.json"
    meta_data = {
        "provider": "vertex",
        "model": "gemini-embedding-001",
        "actual_dimensions": sample_embeddings.shape[1],
        "index_type": "flat",
        "created_at": datetime.now().isoformat(),
    }
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta_data, f, indent=2)

    return {
        "mapping": mapping_path,
        "embeddings": embeddings_path,
        "meta": meta_path,
        "index_dir": mock_index_dir,
    }


@pytest.fixture
def mock_conversation_structure(temp_dir):
    """
    Creates a mock conversation directory structure.

    Returns:
        Path to conversation directory
    """
    conv_dir = temp_dir / "conversation_1"
    conv_dir.mkdir(parents=True, exist_ok=True)

    # Create Conversation.txt
    conv_file = conv_dir / "Conversation.txt"
    conv_file.write_text(
        "From: test@example.com\n"
        "To: recipient@example.com\n"
        "Subject: Test Email\n\n"
        "This is a test email content.\n"
        "It has multiple lines.\n"
    )

    # Create Attachments directory
    attach_dir = conv_dir / "Attachments"
    attach_dir.mkdir(exist_ok=True)

    # Create a sample attachment
    attach_file = attach_dir / "document.txt"
    attach_file.write_text("This is attachment content.")

    return conv_dir


# ============================================================================
# GCP/Vertex AI Mocks
# ============================================================================


@pytest.fixture
def mock_vertex_ai():
    """
    Provides mocked Vertex AI initialization.

    Yields:
        Mock object for vertexai module
    """
    with patch("vertexai.init") as mock_init:
        mock_init.return_value = None
        yield mock_init


@pytest.fixture
def mock_embed_texts():
    """
    Provides mocked embed_texts function.

    Yields:
        Mock function that returns normalized embeddings
    """

    def mock_embed(texts, **_):
        """Mock embedding function that returns random normalized vectors."""
        n = len(texts)
        dim = 768
        embeddings = np.random.randn(n, dim).astype(np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / norms

    with patch("emailops.llm_client_shim.embed_texts", side_effect=mock_embed) as mock:
        yield mock


@pytest.fixture
def mock_vertex_account_validation():
    """
    Provides mocked account validation that always succeeds.

    Yields:
        Mock validation function
    """
    with patch("emailops.llm_runtime.validate_account") as mock_validate:
        mock_validate.return_value = (True, "OK")
        yield mock_validate


@pytest.fixture
def mock_credentials_file(temp_dir):
    """
    Creates a mock credentials JSON file.

    Returns:
        Path to mock credentials file
    """
    creds_dir = temp_dir / "secrets"
    creds_dir.mkdir(exist_ok=True)

    creds_file = creds_dir / "test-credentials.json"
    creds_data = {
        "type": "service_account",
        "project_id": "test-project",
        "private_key_id": "test-key-id",
        "private_key": "-----BEGIN PRIVATE KEY-----\ntest\n-----END PRIVATE KEY-----\n",
        "client_email": "test@test-project.iam.gserviceaccount.com",
        "client_id": "123456789",
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
    }

    with creds_file.open("w") as f:
        json.dump(creds_data, f)

    return creds_file


# ============================================================================
# Environment Variable Mocks
# ============================================================================


@pytest.fixture
def mock_env_vars():
    """
    Provides context manager for mocking environment variables.

    Yields:
        Dictionary to update with environment variables
    """
    original_env = os.environ.copy()
    env_updates = {
        "EMBED_PROVIDER": "vertex",
        "VERTEX_PROJECT": "test-project",
        "GCP_PROJECT": "test-project",
        "GCP_REGION": "us-central1",
        "INDEX_DIRNAME": "_index",
    }
    os.environ.update(env_updates)

    yield env_updates

    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


# ============================================================================
# Logging Mocks
# ============================================================================


@pytest.fixture
def capture_logs(caplog):
    """
    Provides enhanced log capturing with level filtering.

    Args:
        caplog: pytest's built-in caplog fixture

    Returns:
        Caplog fixture with helper methods
    """
    import logging

    caplog.set_level(logging.DEBUG)
    return caplog


# ============================================================================
# Test Data Generators
# ============================================================================


def generate_random_text(length: int = 100) -> str:
    """
    Generates random text for testing.

    Args:
        length: Approximate length of text in words

    Returns:
        Random text string
    """
    import random
    import string

    words = []
    for _ in range(length):
        word_len = random.randint(3, 10)
        word = "".join(random.choices(string.ascii_lowercase, k=word_len))
        words.append(word)

    return " ".join(words)


def generate_embedding(dim: int = 768) -> np.ndarray:
    """
    Generates a random normalized embedding vector.

    Args:
        dim: Embedding dimension

    Returns:
        Normalized embedding vector
    """
    vec = np.random.randn(dim).astype(np.float32)
    return vec / np.linalg.norm(vec)


# ============================================================================
# Assertion Helpers
# ============================================================================


def assert_valid_mapping_entry(entry: dict[str, Any]) -> None:
    """
    Asserts that a mapping entry has all required fields.

    Args:
        entry: Mapping dictionary to validate

    Raises:
        AssertionError: If required fields are missing or invalid
    """
    required_fields = [
        "id",
        "path",
        "conv_id",
        "doc_type",
        "subject",
        "snippet",
        "modified_time",
    ]

    for field in required_fields:
        assert field in entry, f"Missing required field: {field}"
        assert entry[field] is not None, f"Field {field} is None"

    assert entry["doc_type"] in [
        "conversation",
        "attachment",
    ], f"Invalid doc_type: {entry['doc_type']}"

    assert isinstance(entry["snippet"], str), "snippet must be a string"


def assert_normalized_embeddings(
    embeddings: np.ndarray, tolerance: float = 1e-5
) -> None:
    """
    Asserts that embeddings are properly normalized.

    Args:
        embeddings: NumPy array of embeddings
        tolerance: Tolerance for normalization check

    Raises:
        AssertionError: If embeddings are not normalized
    """
    if embeddings.size == 0:
        return

    assert embeddings.ndim == 2, f"Embeddings must be 2D, got shape {embeddings.shape}"

    norms = np.linalg.norm(embeddings, axis=1)
    assert np.allclose(
        norms, 1.0, atol=tolerance
    ), f"Embeddings not normalized: norms range from {norms.min():.4f} to {norms.max():.4f}"


# ============================================================================
# Pytest Hooks
# ============================================================================


def pytest_configure(config):
    """
    Pytest configuration hook.

    Args:
        config: Pytest config object
    """
    # Add custom markers
    config.addinivalue_line("markers", "unit: mark test as a unit test")
    config.addinivalue_line("markers", "integration: mark test as an integration test")
    config.addinivalue_line("markers", "slow: mark test as slow running")


def pytest_collection_modifyitems(items):
    """
    Modify test collection to add markers automatically.

    Args:
        config: Pytest config object
        items: List of test items
    """
    for item in items:
        # Auto-mark integration tests
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        # Auto-mark unit tests
        elif "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
