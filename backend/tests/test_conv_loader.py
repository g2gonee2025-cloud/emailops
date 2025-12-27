import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from cortex.ingestion.conv_loader import (
    _load_conversation_text,
    load_conversation,
    load_summary,
)


class TestConvLoader(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())
        self.convo_dir = self.test_dir / "test_convo"
        self.convo_dir.mkdir()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_load_conversation_text_valid(self):
        f = self.convo_dir / "Conversation.txt"
        f.write_text("Main Body")
        content = _load_conversation_text(self.convo_dir)
        self.assertEqual(content, "Main Body")

    def test_load_conversation_text_fallback(self):
        f = self.convo_dir / "conversation.txt"
        f.write_text("Fallback Body")
        content = _load_conversation_text(self.convo_dir)
        self.assertEqual(content, "Fallback Body")

    @patch("cortex.ingestion.conv_loader.logger")
    def test_load_conversation_missing(self, mock_logger):
        content = _load_conversation_text(self.convo_dir)
        self.assertEqual(content, "")

    def test_load_summary(self):
        f = self.convo_dir / "summary.json"
        f.write_text('{"summary": "test"}', encoding="utf-8-sig")
        data = load_summary(self.convo_dir)
        self.assertEqual(data, {"summary": "test"})

    def test_load_summary_missing(self):
        data = load_summary(self.convo_dir)
        self.assertEqual(data, {})

    @patch("cortex.ingestion.conv_loader.get_config")
    @patch("cortex.ingestion.conv_loader._process_attachments")
    @patch("cortex.ingestion.conv_loader.load_manifest")
    def test_load_conversation_valid(
        self, mock_load_manifest, mock_process_att, mock_config
    ):
        # Setup files
        (self.convo_dir / "Conversation.txt").write_text("Body")
        (self.convo_dir / "manifest.json").write_text("{}")

        # Mocks
        mock_process_att.return_value = ([], "")
        mock_load_manifest.return_value = {"id": 1}

        # Mock Config object to satisfy the new security check
        mock_config_instance = MagicMock()
        mock_config_instance.limits.max_total_attachments_mb = 10
        mock_config_instance.limits.max_attachment_text_chars = 1000
        mock_config_instance.limits.skip_attachment_over_mb = 5.0
        mock_config_instance.directories.export_root = str(self.test_dir)
        mock_config.return_value = mock_config_instance

        result = load_conversation(self.convo_dir)

        self.assertIsNotNone(result)
        self.assertEqual(result["conversation_txt"], "Body")
        self.assertEqual(result["manifest"], {"id": 1})

    @patch("cortex.ingestion.conv_loader.get_config")
    def test_load_conversation_invalid_path(self, mock_config):
        mock_config.return_value.directories.export_root = "/secure_root"
        result = load_conversation(Path("/non/existent"))
        self.assertIsNone(result)

    @patch("cortex.ingestion.conv_loader.get_config")
    def test_load_conversation_not_dir(self, mock_config):
        mock_config.return_value.directories.export_root = str(self.test_dir)
        f = self.test_dir / "file"
        f.touch()
        result = load_conversation(f)
        self.assertIsNone(result)

    @patch("cortex.ingestion.conv_loader.get_config")
    def test_load_conversation_path_traversal(self, mock_config):
        # Configure the secure root directory
        secure_root = self.test_dir.resolve()
        mock_config.return_value.directories.export_root = str(secure_root)

        # Attempt to access a path outside the secure root
        malicious_path = self.test_dir / ".." / ".."
        result = load_conversation(malicious_path)

        # Assert that the operation was blocked (returned None)
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
