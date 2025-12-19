import pytest
from cortex.chunking.chunker import _apply_progressive_scaling


@pytest.mark.parametrize(
    "total_tokens,max_tokens,overlap_tokens,expected_max,expected_overlap",
    [
        # Small text: No scaling (factor 1.0)
        # 1000 tokens, chunk=500, overlap=50
        # est_chunks ~ 2 -> factor 1.0
        (1000, 500, 50, 500, 50),
        # Medium text: 1.5x scaling (factor 1.5)
        # 40,000 tokens, chunk=500, overlap=50 (eff=450)
        # est_chunks ~ 88 -> factor 1.5
        # new_max = 500 * 1.5 = 750
        # new_overlap = 50 * (750/500) = 75
        (40000, 500, 50, 750, 75),
        # Large text: 2.0x scaling (factor 2.0)
        # 100,000 tokens, chunk=500, overlap=50
        # est_chunks ~ 222 -> factor 2.0
        # new_max = 1000
        # new_overlap = 100
        (100000, 500, 50, 1000, 100),
        # Very large text: 3.0x scaling (factor 3.0)
        # 300,000 tokens
        # est_chunks ~ 666 -> factor 3.0
        # new_max = 1500
        (300000, 500, 50, 1500, 150),
    ],
)
def test_progressive_scaling_logic(
    total_tokens, max_tokens, overlap_tokens, expected_max, expected_overlap
):
    new_max, new_overlap = _apply_progressive_scaling(
        total_tokens, max_tokens, overlap_tokens
    )
    assert new_max == expected_max
    assert new_overlap == expected_overlap


def test_progressive_scaling_preserves_min_overlap():
    # Edge case: scaling shouldn't break overlap logic
    new_max, new_overlap = _apply_progressive_scaling(10000, 100, 10)
    assert new_max >= 100
    assert new_overlap >= 10
