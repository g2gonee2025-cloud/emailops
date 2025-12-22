import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parents[2]
sys.path.append(str(root_dir / "backend" / "src"))
import unittest  # noqa: E402
from pathlib import Path  # noqa: E402

# Add src to path
sys.path.append(str(Path.cwd() / "backend" / "src"))

from cortex.chunking.chunker import ChunkingInput, chunk_text  # noqa: E402
from cortex.ingestion.quoted_masks import detect_quoted_spans  # noqa: E402


class TestCorrectness(unittest.TestCase):
    def test_quoted_text_detection_strictness(self):
        """Verify 'From: ' heuristic is stricter."""
        # Case 1: "From: Alice" without @ - Should NOT be quoted (risky heuristic removed)
        text_risky = "From: Alice\nThis is normal text."
        spans_risky = detect_quoted_spans(text_risky)
        # Expectation: No quoted text because "From: Alice" lacks "@"
        # Wait, if I upgraded it, it should return [] or specific spans?
        # If it matches, it marks from index 0 to end.

        # NOTE: This depends on how detect_quoted_spans logic works.
        # If it assumes "From: " starts a quoted block, it covers until end.
        # With my fix, "From: Alice" (no @) should NOT trigger start.

        # But wait, detect_quoted_spans tracks lines.
        # If line is not a quote header, it's normal text.

        # Let's see if we covered "From: Alice <alice@example.com>"
        text_safe = (
            "From: Alice <alice@example.com>\nSent: Yesterday\n\nQuoted content."
        )
        spans_safe = detect_quoted_spans(text_safe)

        # We expect spans_safe to find something.
        # We expect spans_risky to be EMPTY or different from pre-fix.

        print(f"Risky text spans: {spans_risky}")
        self.assertEqual(
            len(spans_risky), 0, "Should not detect plain 'From: Alice' as quote header"
        )

        print(f"Safe text spans: {spans_safe}")
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
                chunk_type_hint="attachment_text",  # Passing string, model converts?
                # Wait, ChunkingInput defines type_hint as optional str.
                # chunk_text logic uses it.
            )
        )

        self.assertEqual(chunks[0].chunk_type, "attachment_text")
        print("Chunk type hint verified.")


if __name__ == "__main__":
    unittest.main()
