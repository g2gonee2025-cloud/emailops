"""
Tests for the 'draft' command.
"""

import argparse
import unittest
from io import StringIO
from unittest.mock import MagicMock, patch

import httpx
from cortex_cli.cmd_draft import run_draft_command


class TestDraftCommand(unittest.TestCase):
    @patch("sys.stdout", new_callable=StringIO)
    @patch("httpx.Client")
    def test_run_draft_command_success(self, mock_client, mock_stdout):
        """
        Test the 'draft' command with a successful API call.
        """
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "draft": {"draft": "This is a test draft."},
            "correlation_id": "test-id",
            "iterations": 1,
        }
        mock_client.return_value.__enter__.return_value.post.return_value = (
            mock_response
        )

        args = argparse.Namespace(
            instruction="test instruction",
            thread_id=None,
            reply_to_message_id=None,
            tone="professional",
        )
        run_draft_command(args)

        output = mock_stdout.getvalue()
        self.assertIn("This is a test draft.", output)
        self.assertIn("Correlation ID: test-id", output)

    @patch("sys.stdout", new_callable=StringIO)
    @patch("httpx.Client")
    def test_run_draft_command_http_error(self, mock_client, mock_stdout):
        """
        Test the 'draft' command with an HTTP error.
        """
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_request = MagicMock()
        mock_client.return_value.__enter__.return_value.post.side_effect = (
            httpx.HTTPStatusError("Error", request=mock_request, response=mock_response)
        )

        args = argparse.Namespace(
            instruction="test instruction",
            thread_id=None,
            reply_to_message_id=None,
            tone="professional",
        )
        run_draft_command(args)

        output = mock_stdout.getvalue()
        self.assertIn("Error: 500 - Internal Server Error", output)


if __name__ == "__main__":
    unittest.main()
