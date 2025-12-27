
import unittest
from cortex.chunking.chunker import ChunkingInput, chunk_text, Span

class ChunkerTest(unittest.TestCase):
    def test_empty_text(self):
        """Test that empty text returns no chunks."""
        inp = ChunkingInput(text="", section_path="test")
        chunks = chunk_text(inp)
        self.assertEqual(len(chunks), 0)

    def test_short_text_no_chunking(self):
        """Test that text smaller than max_tokens results in one chunk."""
        text = "This is a short text that does not need chunking."
        inp = ChunkingInput(text=text, section_path="test", max_tokens=100)
        chunks = chunk_text(inp)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].text, text)
        self.assertEqual(chunks[0].char_start, 0)
        self.assertEqual(chunks[0].char_end, len(text))

    def test_simple_chunking_and_offsets(self):
        """Test basic sliding-window chunking and correct character offsets."""
        text = "This is sentence one. This is sentence two."
        inp = ChunkingInput(
            text=text,
            section_path="test",
            max_tokens=5,
            min_tokens=3,
            overlap_tokens=2,
            preserve_sentences=False
        )
        chunks = chunk_text(inp)

        self.assertEqual(len(chunks), 3)

        self.assertEqual(chunks[0].text, "This is sentence one.")
        self.assertEqual(chunks[0].char_start, 0)
        self.assertEqual(chunks[0].char_end, 21)

        self.assertEqual(chunks[1].text, " one. This is sentence")
        self.assertEqual(chunks[1].char_start, 16)
        self.assertEqual(chunks[1].char_end, 38)

        self.assertEqual(chunks[2].text, " is sentence two.")
        self.assertEqual(chunks[2].char_start, 26)
        self.assertEqual(chunks[2].char_end, 43)


    def test_sentence_boundary(self):
        """Test that chunks are split at sentence boundaries when requested."""
        text = "This is sentence one. This is sentence two. This is the third sentence and it is longer."
        inp = ChunkingInput(text=text, section_path="test", max_tokens=15, min_tokens=5, overlap_tokens=3, preserve_sentences=True)
        chunks = chunk_text(inp)

        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0].text, "This is sentence one. This is sentence two. This is the third sentence")
        self.assertEqual(chunks[1].text, " the third sentence and it is longer.")

    def test_quoted_spans(self):
        """Test that chunks are correctly classified as 'quoted_history'."""
        text = "This is a normal sentence. [QUOTE] This is a quote. [/QUOTE]"
        quoted_spans = [Span(start=29, end=50)]
        inp = ChunkingInput(text=text, section_path="test", max_tokens=8, min_tokens=2, overlap_tokens=2, quoted_spans=quoted_spans)
        chunks = chunk_text(inp)

        chunk_types = [c.chunk_type for c in chunks]
        self.assertIn("message_body", chunk_types)
        self.assertIn("quoted_history", chunk_types)

if __name__ == "__main__":
    unittest.main()
