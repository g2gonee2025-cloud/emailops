import sys
from pathlib import Path

try:
    # Assumes script is in DEV_SCRIPTS_DIR, i.e., three levels deep
    root_dir = Path(__file__).resolve().parents[2]
except IndexError:
    # Fallback for running from root or other locations
    root_dir = Path.cwd()
sys.path.append(str(root_dir / "backend" / "src"))
import unittest

from cortex.chunking.chunker import ChunkingInput, chunk_text
from cortex.ingestion.quoted_masks import detect_quoted_spans


class TestCorrectness(unittest.TestCase):
    def test_quoted_text_detection_strictness(self):
        """Verify 'From: ' heuristic is stricter."""
        # Case 1: "From: Alice" without @ - Should NOT be quoted (risky heuristic removed)
        text_risky = "From: Alice\nThis is normal text."
        spans_risky = detect_quoted_spans(text_risky)
        self.assertEqual(
            len(spans_risky), 0, "Should not detect plain 'From: Alice' as quote header"
        )

        text_safe = (
            "From: Alice <alice@example.com>\nSent: Yesterday\n\nQuoted content."
        )
        spans_safe = detect_quoted_spans(text_safe)
        self.assertGreater(
            len(spans_safe), 0, "Should detect 'From: ... @ ...' as quote header"
        )

    def test_chunk_type_hint(self):
        """Verify chunk_type_hint is propagated."""

        text = "Attachment content here."
        chunks = chunk_text(
            ChunkingInput(
                text=text,
                section_path="attachments/test.txt",
                # NOTE: `chunk_type_hint` is the correct field in the Pydantic model.
                chunk_type_hint="attachment_text",
            )
        )

        self.assertGreater(len(chunks), 0, "Chunking should produce at least one chunk")
        if chunks:
            self.assertEqual(chunks[0].chunk_type, "attachment_text")


if __name__ == "__main__":
    unittest.main()
