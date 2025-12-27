import argparse
from unittest.mock import MagicMock, patch

from cortex_cli.cmd_schema import cmd_schema_check


class TestCmdSchema:
    @patch("cortex_cli.cmd_schema.get_db_session")
    @patch("cortex_cli.cmd_schema.GraphExtractor")
    def test_cmd_schema_check(self, mock_graph_extractor, mock_get_db_session, capsys):
        # Mock the database session
        mock_session = MagicMock()
        mock_get_db_session.return_value.__enter__.return_value = mock_session

        # Mock the results for the execute calls
        mock_conv_ids_result = MagicMock()
        mock_conv_ids_result.scalars.return_value.all.return_value = ["conv1", "conv2"]

        mock_chunks_result1 = MagicMock()
        mock_chunks_result1.scalars.return_value.all.return_value = [
            "chunk1",
            "chunk2",
        ]

        mock_chunks_result2 = MagicMock()
        mock_chunks_result2.scalars.return_value.all.return_value = [
            "chunk3",
            "chunk4",
        ]

        # The first call to execute gets conv_ids, the next two get chunks
        mock_session.execute.side_effect = [
            mock_conv_ids_result,
            mock_chunks_result1,
            mock_chunks_result2,
        ]

        # Mock the GraphExtractor
        mock_extractor_instance = MagicMock()
        mock_graph_extractor.return_value = mock_extractor_instance
        mock_g = MagicMock()
        mock_g.nodes.return_value = [
            ("node1", {"type": "PERSON"}),
            ("node2", {"type": "ORGANIZATION"}),
        ]
        mock_g.edges.return_value = [
            ("node1", "node2", {"relation": "WORKS_FOR"}),
        ]
        mock_extractor_instance.extract_graph.return_value = mock_g

        # Create a mock argparse.Namespace object
        args = argparse.Namespace(limit=2)

        # Call the function
        cmd_schema_check(args)

        # Capture the output
        captured = capsys.readouterr()
        output = captured.out + captured.err

        # Assert that the output is as expected
        assert "Fetching up to 2 random conversations..." in output
        assert "Extracting graphs sequentially..." in output
        assert "Processing text 1/2" in output
        assert "Processing text 2/2" in output
        assert "=== FINAL SCHEMA REPORT ===" in output
        assert "Top Node Types:" in output
        assert "Top Relations:" in output
