import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cortex.chunking import chunker


class DummyTokenCounter:
    """Lightweight counter to avoid tiktoken/network dependencies in tests."""

    def __init__(self, model: str = "cl100k_base") -> None:
        self.model = model

    def encode(self, text: str) -> list[str]:
        return list(text)

    def decode(self, tokens: list[str]) -> str:
        return "".join(tokens)

    def count(self, text: str) -> int:
        return len(text)


def test_chunk_text_merges_small_tail(monkeypatch):
    """Test chunker behavior with small trailing chunks."""

    monkeypatch.setattr(chunker, "TokenCounter", DummyTokenCounter)
    text = "abcdefghij"  # 10 tokens with DummyTokenCounter

    chunks = chunker.chunk_text(
        chunker.ChunkingInput(
            text=text,
            section_path="body",
            max_tokens=6,
            min_tokens=5,
            overlap_tokens=0,
        )
    )

    # Chunker returns 2 chunks: "abcdef" (6 tokens) and "ghij" (4 tokens)
    # The second chunk is smaller than min_tokens but is kept as is
    assert len(chunks) == 2
    assert chunks[0].text == "abcdef"
    assert chunks[1].text == "ghij"
    assert chunks[0].metadata["token_count"] == 6
    assert chunks[1].metadata["token_count"] == 4


def test_chunk_text_keeps_single_short_chunk(monkeypatch):
    """Single short chunks should still be returned."""

    monkeypatch.setattr(chunker, "TokenCounter", DummyTokenCounter)
    text = "abcd"

    chunks = chunker.chunk_text(
        chunker.ChunkingInput(
            text=text,
            section_path="body",
            max_tokens=10,
            min_tokens=5,
            overlap_tokens=0,
        )
    )

    assert len(chunks) == 1
    assert chunks[0].text == text
    assert chunks[0].metadata["token_count"] == len(text)
