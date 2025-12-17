import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cortex.chunking import chunker


class DummyTokenCounter:
    """Lightweight counter to avoid tiktoken/network dependencies in tests."""

    def __init__(self, model: str = "cl100k_base") -> None:  # noqa: ARG002
        self.model = model

    def encode(self, text: str) -> list[str]:
        return list(text)

    def decode(self, tokens: list[str]) -> str:
        return "".join(tokens)


def test_chunk_text_merges_small_tail(monkeypatch):
    """Ensure trailing chunks smaller than min_tokens merge with the previous chunk."""

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

    assert len(chunks) == 1
    assert chunks[0].text == text
    assert chunks[0].metadata["token_count"] == len(text)
    assert chunks[0].char_start == 0
    assert chunks[0].char_end == len(text)


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
