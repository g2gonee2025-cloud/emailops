# Make imports work as if running from the project root
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.append(str(Path(__file__).resolve().parents[3] / "backend/src"))

from cortex.safety.grounding import (
    _cosine_similarity,
    check_grounding_embedding,
    extract_claims_simple,
)

# --- Fixtures ---


@pytest.fixture
def mock_embeddings_client():
    """Mocks the EmbeddingsClient to avoid actual model loading and API calls."""
    with patch("cortex.safety.grounding.get_embeddings_client") as mock_get_client:
        mock_client_instance = MagicMock()

        # Simple embedding function for predictable testing
        def mock_embed(text: str) -> list[float]:
            # Check for "unsupported" first, as it contains "supported"
            if "unsupported" in text.lower():
                return [0.0, 1.0, 0.0]
            if "supported" in text.lower():
                return [1.0, 0.0, 0.0]
            if "partially" in text.lower():
                return [0.8, 0.2, 0.0]
            return [0.0, 0.0, 1.0]  # Default vector

        def mock_embed_texts(texts: list[str]) -> list[list[float]]:
            return [mock_embed(text) for text in texts]

        mock_client_instance.embed.side_effect = mock_embed
        mock_client_instance.embed_texts.side_effect = mock_embed_texts
        mock_get_client.return_value = mock_client_instance
        yield mock_client_instance


# --- Test Cases ---


def test_cosine_similarity():
    """Test the cosine similarity calculation."""
    assert _cosine_similarity([1, 0], [1, 0]) == pytest.approx(1.0)
    assert _cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)
    assert _cosine_similarity([1, 0], [-1, 0]) == pytest.approx(-1.0)
    assert _cosine_similarity([1, 2, 3], [4, 5, 6]) == pytest.approx(0.9746318)
    assert _cosine_similarity([0, 0], [1, 1]) == pytest.approx(
        0.0
    ), "Similarity with a zero vector should be 0"


def test_extract_claims_simple():
    """Test the heuristic-based claim extraction."""
    text = (
        "This is a factual statement. Is this a question? "
        "This might be a hedged statement. "
        "This is another supported fact. Short one. "
        "Based on the context, this is a meta-statement."
    )
    claims = extract_claims_simple(text)
    assert claims == [
        "This is a factual statement",
        "This is another supported fact",
    ]


def test_grounding_no_claims_is_not_grounded(mock_embeddings_client):
    """
    CRITICAL: Test that an answer with no verifiable claims is NOT considered grounded.
    This is a safety-critical correction from the original logic.
    """
    answer = "I don't know."  # This will result in no claims being extracted
    facts = ["Some random fact."]

    result = check_grounding_embedding(answer, facts)

    assert not result.is_grounded
    assert result.confidence == pytest.approx(1.0)
    assert result.grounding_ratio == pytest.approx(0.0)
    assert len(result.claim_analyses) == 0


def test_grounding_with_claims_but_no_facts(mock_embeddings_client):
    """Test that an answer is not grounded if there are no facts to support it."""
    answer = "This is a fully supported statement."
    facts = []

    result = check_grounding_embedding(answer, facts)

    assert not result.is_grounded
    assert result.confidence == pytest.approx(1.0)  # Should be certain it's not grounded
    assert result.grounding_ratio == pytest.approx(0.0)
    assert len(result.unsupported_claims) == 1
    assert result.unsupported_claims[0] == "This is a fully supported statement"


def test_grounding_fully_supported(mock_embeddings_client):
    """Test a scenario where all claims are supported by facts."""
    answer = "This is a supported claim. This is also a supported statement."
    facts = ["Fact about a supported claim."]

    result = check_grounding_embedding(
        answer, facts, support_threshold=0.9, grounding_threshold=0.7
    )

    assert result.is_grounded
    assert result.confidence > 0.9
    assert result.grounding_ratio == pytest.approx(1.0)
    assert len(result.unsupported_claims) == 0
    assert all(c.is_supported for c in result.claim_analyses)


def test_grounding_fully_unsupported(mock_embeddings_client):
    """Test a scenario where no claims are supported."""
    answer = "This is an unsupported claim."
    facts = ["Fact about something else entirely."]

    result = check_grounding_embedding(answer, facts, support_threshold=0.9)

    assert not result.is_grounded
    assert result.confidence < 0.1
    assert result.grounding_ratio == pytest.approx(0.0)
    assert len(result.unsupported_claims) == 1


def test_grounding_partially_supported_and_grounded(mock_embeddings_client):
    """Test when enough claims are supported to pass the grounding threshold."""
    answer = "This is a supported claim. This is another supported one. This is an unsupported one."
    facts = ["Fact that is very much about a supported claim."]
    # Expect 3 claims to be extracted. 2 should be supported.

    result = check_grounding_embedding(
        answer, facts, support_threshold=0.9, grounding_threshold=0.6
    )

    assert result.is_grounded
    assert len(result.claim_analyses) == 3
    assert sum(1 for c in result.claim_analyses if c.is_supported) == 2
    assert len(result.unsupported_claims) == 1


def test_grounding_partially_supported_and_not_grounded(mock_embeddings_client):
    """Test when not enough claims are supported to pass the grounding threshold."""
    answer = "This is a supported claim. This is an unsupported one. This is another unsupported one."
    facts = ["Fact that is very much about a supported claim."]
    # Expect 3 claims to be extracted. 1 should be supported.

    result = check_grounding_embedding(
        answer, facts, support_threshold=0.9, grounding_threshold=0.7
    )

    assert not result.is_grounded
    assert len(result.claim_analyses) == 3
    assert sum(1 for c in result.claim_analyses if c.is_supported) == 1
    assert len(result.unsupported_claims) == 2
