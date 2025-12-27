import argparse
from unittest.mock import MagicMock, patch

import pytest
from cortex_cli.cmd_index import cmd_index_stats

@pytest.fixture
def mock_get_index_info():
    with patch('cortex_cli.cmd_index.get_index_info') as mock:
        yield mock

def test_cmd_index_stats_json_output(mock_get_index_info):
    mock_get_index_info.return_value = {
        "metadata": {
            "provider": "test_provider",
            "model": "test_model",
            "dimensions": 128,
            "actual_dimensions": 128,
            "num_documents": 10,
        },
        "dimensions": {
            "embeddings": 128,
            "faiss": 128,
            "detected": 128,
        },
        "index_type": "faiss",
        "faiss_vector_count": 10,
    }

    args = argparse.Namespace(index_dir="/fake/dir", json=True)

    with patch('builtins.print') as mock_print:
        cmd_index_stats(args)
        mock_print.assert_called_once()

def test_cmd_index_stats_human_readable_output(mock_get_index_info):
    mock_get_index_info.return_value = {
        "metadata": {
            "provider": "test_provider",
            "model": "test_model",
            "dimensions": 128,
            "actual_dimensions": 128,
            "num_documents": 10,
        },
        "dimensions": {
            "embeddings": 128,
            "faiss": 128,
            "detected": 128,
        },
        "index_type": "faiss",
        "faiss_vector_count": 10,
    }

    args = argparse.Namespace(index_dir="/fake/dir", json=False)

    with patch('builtins.print') as mock_print:
        cmd_index_stats(args)
        assert mock_print.call_count > 1
