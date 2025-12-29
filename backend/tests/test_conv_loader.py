import shutil
import tempfile
import unittest
from pathlib import Path

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

    def test_load_conversation_missing(self):
        """Test loading conversation from directory with no conversation file."""
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

    @unittest.skip(
        "Requires EXPORT_ROOT config override - security test should use production config"
    )
    def test_load_conversation_valid(self):
        """Test loading a valid conversation with real file processing."""
        import json
        import os

        from cortex.config.loader import reset_config

        # Setup files
        (self.convo_dir / "Conversation.txt").write_text("Body")
        manifest_data = {"id": 1, "subject": "Test", "sender": "test@example.com"}
        (self.convo_dir / "manifest.json").write_text(json.dumps(manifest_data))

        # Set export_root via environment variable for config
        old_env = os.environ.get("EXPORT_ROOT")
        os.environ["EXPORT_ROOT"] = str(self.test_dir)
        reset_config()  # Clear cached config so new env var is picked up

        try:
            result = load_conversation(self.convo_dir)

            self.assertIsNotNone(result)
            self.assertEqual(result["conversation_txt"], "Body")
            self.assertEqual(result["manifest"]["id"], 1)
        finally:
            if old_env is not None:
                os.environ["EXPORT_ROOT"] = old_env
            elif "EXPORT_ROOT" in os.environ:
                del os.environ["EXPORT_ROOT"]
            reset_config()  # Restore original config

    def test_load_conversation_invalid_path(self):
        """Test loading from non-existent path returns None."""
        import os

        os.environ["EXPORT_ROOT"] = "/secure_root"
        try:
            result = load_conversation(Path("/non/existent"))
            self.assertIsNone(result)
        finally:
            del os.environ["EXPORT_ROOT"]

    def test_load_conversation_not_dir(self):
        """Test loading from file (not directory) returns None."""
        import os

        os.environ["EXPORT_ROOT"] = str(self.test_dir)
        try:
            f = self.test_dir / "file"
            f.touch()
            result = load_conversation(f)
            self.assertIsNone(result)
        finally:
            del os.environ["EXPORT_ROOT"]

    def test_load_conversation_path_traversal(self):
        """Test that path traversal attempts are blocked."""
        import os

        secure_root = self.test_dir.resolve()
        os.environ["EXPORT_ROOT"] = str(secure_root)
        try:
            # Attempt to access a path outside the secure root
            malicious_path = self.test_dir / ".." / ".."
            result = load_conversation(malicious_path)
            # Assert that the operation was blocked (returned None)
            self.assertIsNone(result)
        finally:
            del os.environ["EXPORT_ROOT"]


if __name__ == "__main__":
    unittest.main()
