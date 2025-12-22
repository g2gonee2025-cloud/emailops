from unittest.mock import MagicMock, patch

from cortex.orchestration.nodes import (
    _extract_evidence_from_answer,
    extract_document_mentions,
    node_assemble_context,
    node_classify_query,
    node_generate_answer,
    node_prepare_draft_query,
)
from cortex.retrieval.results import SearchResultItem, SearchResults


class TestNodesUnit:
    def test_extract_document_mentions(self):
        text = "Please review report.pdf and invoice.docx. Also 'Project Plan' is relevant."
        mentions = extract_document_mentions(text)
        assert "report.pdf" in mentions
        assert "invoice.docx" in mentions
        assert "Project Plan" in mentions

    def test_extract_document_mentions_empty(self):
        assert extract_document_mentions("") == []
        assert extract_document_mentions("Hello world") == []

    def test_node_assemble_context(self):
        results = SearchResults(
            query="test",
            results=[
                SearchResultItem(
                    chunk_id="c1", highlights=["This is line 1."], score=0.9
                ),
                SearchResultItem(
                    chunk_id="m1", highlights=["This is an email."], score=0.8
                ),
            ],
            reranker="test",
        )
        state = {"retrieval_results": results}
        output = node_assemble_context(state)
        context = output["assembled_context"]

        assert "[Source 1 (ID: c1)]" in context
        assert "This is line 1." in context
        assert "[Source 2 (ID: m1)]" in context
        assert "This is an email." in context

    def test_node_assemble_context_empty(self):
        state = {"retrieval_results": None}
        assert node_assemble_context(state)["assembled_context"] == ""

    def test_extract_evidence_from_answer(self):
        answer = "Based on [Source 1], the project is done. Also see (Source 2)."
        results = SearchResults(
            query="test",
            results=[
                SearchResultItem(
                    chunk_id="c1", highlights=["Project is done"], score=1.0
                ),
                SearchResultItem(chunk_id="m1", highlights=["Confirming"], score=0.9),
            ],
            reranker="test",
        )
        evidence = _extract_evidence_from_answer(answer, results)

        assert len(evidence) == 2
        assert evidence[0].chunk_id == "c1"
        assert evidence[1].chunk_id == "m1"

    @patch("cortex.orchestration.nodes.complete_text")
    def test_node_generate_answer(self, mock_complete):
        mock_complete.return_value = "The answer is 42 [Source 1]."

        state = {
            "query": "What is the answer?",
            "assembled_context": "Context here",
            "graph_context": "Graph here",
            "retrieval_results": SearchResults(
                query="test",
                results=[SearchResultItem(chunk_id="c1", highlights=["42"], score=1.0)],
                reranker="test",
            ),
        }

        output = node_generate_answer(state)
        answer = output["answer"]

        assert answer.answer_markdown == "The answer is 42 [Source 1]."
        assert len(answer.evidence) == 1
        assert answer.evidence[0].chunk_id == "c1"
        assert answer.confidence_overall > 0.5

    @patch("cortex.orchestration.nodes.tool_classify_query")
    def test_node_classify_query(self, mock_tool):
        mock_tool.return_value = MagicMock(type="semantic")
        state = {"query": "test query"}

        output = node_classify_query(state)
        assert output["classification"].type == "semantic"

    def test_node_prepare_draft_query(self):
        state = {"explicit_query": "Write an email"}
        output = node_prepare_draft_query(state)
        assert output["query"] == "Write an email"
