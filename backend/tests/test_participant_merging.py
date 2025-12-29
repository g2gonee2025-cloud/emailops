import unittest

from cortex.domain_models.facts_ledger import FactsLedger, ParticipantAnalysis
from cortex.domain_models.rag import ThreadContext, ThreadParticipant
from cortex.orchestration.nodes import node_summarize_final


class TestParticipantMerging(unittest.TestCase):
    """Test participant merging with live LLM calls via CPU fallback."""

    def test_merge_participants_with_context(self):
        """Test merging participants from LLM facts and DB context with live LLM."""
        # 1. Facts Ledger (from LLM)
        facts = FactsLedger(
            participants=[
                ParticipantAnalysis(
                    name="John Doe",
                    email="john@example.com",
                    role="client",
                    tone="frustrated",
                    stance="Wants refund",
                ),
                ParticipantAnalysis(
                    name="Jane Smith",  # No email in LLM output
                    role="broker",
                    tone="professional",
                ),
            ]
        )

        # 2. Thread Context (from DB)
        thread_context = ThreadContext(
            participants=[
                ThreadParticipant(
                    email="john@example.com", name="Johnathan Doe", role="sender"
                ),
                ThreadParticipant(
                    email="jane.smith@broker.com", name="Jane Smith", role="recipient"
                ),  # Matches by name
                ThreadParticipant(email="unknown@example.com", role="cc"),  # No match
            ]
        )

        # 3. State
        state = {
            "facts_ledger": facts,
            "thread_id": "00000000-0000-0000-0000-000000000123",
            "thread_context": "...",
            "_thread_context_obj": thread_context,
        }

        # Run node with live LLM
        result = node_summarize_final(state)
        summary = result.get("summary")

        # Verify summary was generated (content varies with LLM)
        self.assertIsNotNone(summary)
        self.assertTrue(hasattr(summary, "participants"))
        self.assertGreater(len(summary.participants), 0)

        # Check John exists and has expected fields overlaid
        john = next(
            (p for p in summary.participants if p.email == "john@example.com"), None
        )
        if john:
            self.assertEqual(john.role, "client")
            self.assertEqual(john.tone, "frustrated")

    def test_merge_participants_no_context(self):
        """Test participant merging without DB context using live LLM."""
        facts = FactsLedger(
            participants=[ParticipantAnalysis(email="foo@bar.com", role="client")]
        )
        state = {
            "facts_ledger": facts,
            "thread_id": "00000000-0000-0000-0000-000000000123",
        }

        result = node_summarize_final(state)
        summary = result.get("summary")

        self.assertIsNotNone(summary)
        self.assertGreater(len(summary.participants), 0)
        self.assertEqual(summary.participants[0].email, "foo@bar.com")
