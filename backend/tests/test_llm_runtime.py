import unittest

from cortex.llm.runtime import _try_load_json


class TestLLMRuntime(unittest.TestCase):
    def test_extract_json_simple(self):
        """Test simple JSON object extraction."""
        text = '{"key": "value"}'
        self.assertEqual(_try_load_json(text), {"key": "value"})

    def test_extract_json_markdown(self):
        """Test extraction from markdown code blocks."""
        text = 'Here is the JSON:\n```json\n{"key": "value"}\n```'
        self.assertEqual(_try_load_json(text), {"key": "value"})

    def test_extract_json_markdown_no_lang(self):
        """Test extraction from markdown without language specifier."""
        text = '```\n{"key": "value"}\n```'
        self.assertEqual(_try_load_json(text), {"key": "value"})

    def test_extract_json_with_chatter(self):
        """Test extraction with conversational chatter."""
        text = 'Sure, here is the JSON you requested:\n\n{"key": "value"}\n\nHope that helps!'
        self.assertEqual(_try_load_json(text), {"key": "value"})


if __name__ == "__main__":
    unittest.main()
