import argparse
from unittest.mock import MagicMock, patch

from cortex_cli.cmd_rechunk import cmd_db_rechunk


def test_cmd_db_rechunk():
    """Test that the rechunk command calls the backend with the correct arguments."""
    with patch("cortex_cli.cmd_rechunk.run_rechunk") as mock_run_rechunk:
        mock_run_rechunk.return_value = {}  # Mock the return value

        args = argparse.Namespace(
            tenant_id="test-tenant",
            chunk_size_limit=5000,
            dry_run=True,
            max_tokens=1000,
        )
        cmd_db_rechunk(args)

        mock_run_rechunk.assert_called_once_with(
            tenant_id="test-tenant",
            chunk_size_limit=5000,
            dry_run=True,
            max_tokens=1000,
            progress_callback=mock_run_rechunk.call_args.kwargs['progress_callback'],
        )
