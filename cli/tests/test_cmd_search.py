from __future__ import annotations

import argparse
import json
from unittest.mock import MagicMock, patch

from cortex_cli.cmd_search import _run_search_command


def test_search_command_success(capsys):
    """Test the search command with a successful API call."""
    # Mock the httpx response
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "results": [
            {
                "score": 0.9,
                "content": "This is a test result.",
                "chunk_id": "test-chunk-1",
                "metadata": {"chunk_type": "test"},
            }
        ],
        "query_time_ms": 123.45,
    }

    with patch("httpx.Client") as mock_client:
        mock_client.return_value.__enter__.return_value.post.return_value = (
            mock_response
        )

        args = argparse.Namespace(
            query="test query",
            top_k=10,
            tenant="default",
            debug=False,
            json=False,
            fusion="rrf",
        )
        _run_search_command(args)

        captured = capsys.readouterr()
        assert "Found 1 result(s)" in captured.out
        assert "This is a test result." in captured.out
        mock_client.return_value.__enter__.return_value.post.assert_called_with(
            "/search",
            json={
                "query": "test query",
                "k": 10,
                "filters": {},
                "fusion_method": "rrf",
            },
            timeout=60,
        )


def test_search_command_json_output(capsys):
    """Test the search command with the --json flag."""
    # Mock the httpx response
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "results": [],
        "query_time_ms": 50.0,
    }

    with patch("httpx.Client") as mock_client:
        mock_client.return_value.__enter__.return_value.post.return_value = (
            mock_response
        )

        args = argparse.Namespace(
            query="json test",
            top_k=5,
            tenant="default",
            debug=False,
            json=True,
            fusion="weighted_sum",
        )
        _run_search_command(args)

        captured = capsys.readouterr()
        parsed_output = json.loads(captured.out)
        assert parsed_output["query_time_ms"] == 50.0
        assert parsed_output["results"] == []


def test_search_command_http_error(capsys):
    """Test the search command with a 500 HTTP error."""
    with patch("httpx.Client") as mock_client, patch("sys.exit") as mock_exit:
        mock_client.return_value.__enter__.return_value.post.side_effect = Exception(
            "Server Error"
        )

        args = argparse.Namespace(
            query="http error",
            top_k=10,
            tenant="default",
            debug=False,
            json=False,
            fusion="rrf",
        )
        _run_search_command(args)

        captured = capsys.readouterr()
        assert "ERROR: Server Error" in captured.out
        mock_exit.assert_called_with(1)
