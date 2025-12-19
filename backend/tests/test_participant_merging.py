import unittest
from unittest.mock import patch

from cortex.domain_models.facts_ledger import FactsLedger, ParticipantAnalysis
from cortex.domain_models.rag import ThreadContext, ThreadParticipant
from cortex.orchestration.nodes import node_summarize_final


class TestParticipantMerging(unittest.TestCase):
    @patch("cortex.orchestration.nodes.complete_text")
    def test_merge_participants_with_context(self, mock_complete):
        # Mock LLM response
        mock_complete.return_value = "Summary text"

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

        # Run node
        result = node_summarize_final(state)
        summary = result.get("summary")

        # Verify participants
        self.assertIsNotNone(summary)
        participants = summary.participants
        self.assertEqual(len(participants), 3)  # Should have all 3 from DB context

        # Check John (Matched by email)
        john = next(p for p in participants if p.email == "john@example.com")
        self.assertEqual(john.role, "client")  # Overlaid from LLM
        self.assertEqual(john.tone, "frustrated")  # Overlaid
        self.assertEqual(john.stance, "Wants refund")  # Overlaid
        self.assertEqual(john.name, "John Doe")  # LLM name preferred

        # Check Jane (Matched by name)
        jane = next(p for p in participants if p.email == "jane.smith@broker.com")
        self.assertEqual(jane.role, "broker")  # Overlaid
        self.assertEqual(jane.tone, "professional")  # Overlaid
        self.assertEqual(jane.name, "Jane Smith")

        # Check Unknown (No match)
        unknown = next(p for p in participants if p.email == "unknown@example.com")
        self.assertEqual(unknown.role, "other")  # Default
        self.assertEqual(unknown.tone, "neutral")

    @patch("cortex.orchestration.nodes.complete_text")
    def test_merge_participants_no_context(self, mock_complete):
        mock_complete.return_value = "Summary"
        facts = FactsLedger(
            participants=[ParticipantAnalysis(email="foo@bar.com", role="client")]
        )
        state = {
            "facts_ledger": facts,
            "thread_id": "00000000-0000-0000-0000-000000000123",
        }

        result = node_summarize_final(state)
        summary = result.get("summary")

        self.assertEqual(len(summary.participants), 1)
        self.assertEqual(summary.participants[0].email, "foo@bar.com")
