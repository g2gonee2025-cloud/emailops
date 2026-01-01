import os
import shutil
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path

from cortex.config.loader import reset_config
from cortex.ingestion.conv_loader import (
    _load_conversation_text,
    load_conversation,
    load_summary,
)


@contextmanager
def _override_export_root(value: str):
    old_prefixed = os.environ.get("OUTLOOKCORTEX_EXPORT_ROOT")
    old_legacy = os.environ.get("EXPORT_ROOT")
    os.environ["OUTLOOKCORTEX_EXPORT_ROOT"] = value
    os.environ["EXPORT_ROOT"] = value
    reset_config()
    try:
        yield
    finally:
        if old_prefixed is not None:
            os.environ["OUTLOOKCORTEX_EXPORT_ROOT"] = old_prefixed
        else:
            os.environ.pop("OUTLOOKCORTEX_EXPORT_ROOT", None)
        if old_legacy is not None:
            os.environ["EXPORT_ROOT"] = old_legacy
        else:
            os.environ.pop("EXPORT_ROOT", None)
        reset_config()


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

    def test_load_conversation_valid(self):
        """Test loading a valid conversation with real file processing."""
        import json

        # Setup files
        (self.convo_dir / "Conversation.txt").write_text("Body")
        manifest_data = {"id": 1, "subject": "Test", "sender": "test@example.com"}
        (self.convo_dir / "manifest.json").write_text(json.dumps(manifest_data))

        with _override_export_root(str(self.test_dir)):
            result = load_conversation(self.convo_dir)

            self.assertIsNotNone(result)
            self.assertEqual(result["conversation_txt"], "Body")
            self.assertEqual(result["manifest"]["id"], 1)

    def test_load_conversation_invalid_path(self):
        """Test loading from non-existent path returns None."""
        with _override_export_root("/secure_root"):
            result = load_conversation(Path("/non/existent"))
            self.assertIsNone(result)

    def test_load_conversation_not_dir(self):
        """Test loading from file (not directory) returns None."""
        with _override_export_root(str(self.test_dir)):
            f = self.test_dir / "file"
            f.touch()
            result = load_conversation(f)
            self.assertIsNone(result)

    def test_load_conversation_path_traversal(self):
        """Test that path traversal attempts are blocked."""
        secure_root = self.test_dir.resolve()
        with _override_export_root(str(secure_root)):
            # Attempt to access a path outside the secure root
            malicious_path = self.test_dir / ".." / ".."
            result = load_conversation(malicious_path)
            # Assert that the operation was blocked (returned None)
            self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
