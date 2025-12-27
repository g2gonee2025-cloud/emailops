from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from cortex.safety.grounding import (
    ClaimAnalysis,
    GroundingCheck,
    GroundingCheckInput,
    _cosine_similarity,
    check_claim_against_facts_embedding,
    check_claim_against_facts_keyword,
    check_grounding_embedding,
    extract_claims_simple,
    is_answer_grounded,
    tool_check_grounding,
)


class TestGrounding(unittest.TestCase):
    def test_extract_claims_simple(self):
        text = "The sky is blue. This is a fact. Is it raining? I think it might rain. This statement is based on the context provided."
        claims = extract_claims_simple(text)
        self.assertEqual(claims, ["The sky is blue"])

    def test_cosine_similarity(self):
        v1 = [1, 0]
        v2 = [0, 1]
        self.assertAlmostEqual(_cosine_similarity(v1, v2), 0.0)

        v1 = [1, 1]
        v2 = [1, 1]
        self.assertAlmostEqual(_cosine_similarity(v1, v2), 1.0)

    def test_check_claim_against_facts_keyword(self):
        claim = "The sky is blue"
        facts = ["The sky has a blue color.", "The grass is green."]
        is_supported, confidence, supporting_fact, method = check_claim_against_facts_keyword(claim, facts)
        self.assertTrue(is_supported)
        self.assertGreater(confidence, 0.0)
        self.assertEqual(supporting_fact, "The sky has a blue color.")
        self.assertEqual(method, "keyword")

    @patch("cortex.embeddings.client.EmbeddingsClient")
    def test_check_claim_against_facts_embedding(self, MockEmbeddingsClient):
        mock_client_instance = MockEmbeddingsClient.return_value
        mock_client_instance.embed.return_value = [0.8, 0.2]
        mock_client_instance.embed_batch.return_value = [[0.8, 0.2], [0.2, 0.8]]

        claim = "The sky is blue"
        facts = ["The sky has a blue color.", "The grass is green."]
        is_supported, confidence, supporting_fact, method = check_claim_against_facts_embedding(claim, facts)

        self.assertTrue(is_supported)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertEqual(supporting_fact, "The sky has a blue color.")
        self.assertEqual(method, "embedding")

    @patch("cortex.safety.grounding.check_grounding_embedding")
    def test_tool_check_grounding_embedding(self, mock_check_grounding_embedding):
        mock_check_grounding_embedding.return_value = GroundingCheck(
            answer_candidate="The sky is blue.",
            is_grounded=True,
            confidence=0.9,
            unsupported_claims=[],
            claim_analyses=[],
            grounding_ratio=1.0,
            method="embedding",
        )
        args = GroundingCheckInput(
            answer_candidate="The sky is blue.",
            facts=["The sky is blue."],
            use_llm=False,
        )
        result = tool_check_grounding(args)
        self.assertTrue(result.is_grounded)
        self.assertEqual(result.method, "embedding")

    def test_is_answer_grounded(self):
        answer = "The sky is blue."
        facts = ["The color of the sky is blue."]
        self.assertTrue(is_answer_grounded(answer, facts))

    def test_empty_answer(self):
        result = tool_check_grounding(GroundingCheckInput(answer_candidate="", facts=["fact"]))
        self.assertTrue(result.is_grounded)
        self.assertEqual(result.confidence, 1.0)
        self.assertEqual(result.method, "not_applicable")

    def test_no_facts_embedding(self):
        result = check_grounding_embedding("This is a claim.", [])
        self.assertFalse(result.is_grounded)
        self.assertEqual(len(result.unsupported_claims), 1)
        self.assertEqual(result.grounding_ratio, 0.0)

    def test_no_claims_embedding(self):
        result = check_grounding_embedding("I think this might be true.", ["fact"])
        self.assertTrue(result.is_grounded)
        self.assertEqual(result.grounding_ratio, 1.0)


if __name__ == "__main__":
    unittest.main()
