import shutil
import tempfile
import time
import unittest
from pathlib import Path

from cortex.text_extraction import _extraction_cache, extract_text


class TestTextExtraction(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())
        self.test_file = self.test_dir / "test.txt"
        self.test_file.write_text("Hello World")

        # Clear cache before each test
        _extraction_cache.clear()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_caching_behavior(self):
        """Test that extraction results are cached."""
        # First call
        text1 = extract_text(self.test_file)
        self.assertEqual(text1, "Hello World")

        # Verify it's in cache
        cache_key = (self.test_file.resolve(), None)
        self.assertIn(cache_key, _extraction_cache)

        # Modify file
        time.sleep(
            1.1
        )  # Wait to ensure mtime changes (some systems have 1s resolution)
        self.test_file.write_text("Modified Content")

        # Second call should get new content because mtime changed
        text2 = extract_text(self.test_file)
        self.assertEqual(text2, "Modified Content")

    def test_iso_date_parsing_logic_mock(self):
        pass


if __name__ == "__main__":
    unittest.main()
