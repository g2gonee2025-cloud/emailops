from unittest.mock import AsyncMock, MagicMock, patch

from cortex.orchestration.nodes import (
    _extract_entity_mentions,
    _extract_evidence_from_answer,
    extract_document_mentions,
    node_assemble_context,
    node_classify_query,
    node_generate_answer,
    node_handle_error,
    node_prepare_draft_query,
)
from cortex.retrieval.results import SearchResultItem, SearchResults


class TestNodesUnit:
    """Basic node tests."""

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


class TestExtractEntityMentions:
    """Test entity mention extraction."""

    def test_extract_empty_text(self):
        """Test empty text returns empty list."""
        assert _extract_entity_mentions("") == []

    def test_extract_quoted_strings(self):
        """Test extracting quoted strings."""
        text = "Find emails about \"Project Alpha\" and 'Meeting Notes'"
        mentions = _extract_entity_mentions(text)
        assert "Project Alpha" in mentions
        assert "Meeting Notes" in mentions

    def test_extract_capitalized_phrases(self):
        """Test extracting capitalized multi-word phrases."""
        text = "What about John Smith and Acme Corporation?"
        mentions = _extract_entity_mentions(text)
        assert "John Smith" in mentions
        assert "Acme Corporation" in mentions

    def test_extract_single_capitalized_words(self):
        """Test extracting single capitalized words."""
        text = "Contact Lisa about this project"
        mentions = _extract_entity_mentions(text)
        assert "Lisa" in mentions

    def test_filters_stop_words(self):
        """Test that common stop words are filtered."""
        text = "What is the status?"
        mentions = _extract_entity_mentions(text)
        # None of these should be in the results
        assert "What" not in mentions
        assert "The" not in mentions

    def test_mixed_extraction(self):
        """Test extraction from mixed patterns."""
        text = "Ask Sarah about 'Budget Report' from Finance Department"
        mentions = _extract_entity_mentions(text)
        assert "Sarah" in mentions
        assert "Budget Report" in mentions
        assert "Finance Department" in mentions


class TestNodeHandleError:
    """Test error handling node."""

    @patch("cortex.orchestration.nodes.log_audit_event", new_callable=AsyncMock)
    async def test_handle_error_logs_and_returns(self, mock_audit):
        """Test error handling logs and returns error state."""
        state = {
            "error": "Something went wrong",
            "tenant_id": "test-tenant",
            "user_id": "user-123",
            "query": "test query",
        }

        result = await node_handle_error(state)

        assert result["error"] == "Something went wrong"
        mock_audit.assert_called_once()

    @patch("cortex.orchestration.nodes.log_audit_event", new_callable=AsyncMock)
    async def test_handle_error_default_error(self, mock_audit):
        """Test error handling with no error in state."""
        state = {}  # No error provided

        result = await node_handle_error(state)

        assert result["error"] == "Unknown error"

    @patch("cortex.orchestration.nodes.log_audit_event", new_callable=AsyncMock)
    async def test_handle_error_audit_failure_is_caught(self, mock_audit):
        """Test that audit logging failure doesn't crash the handler."""
        mock_audit.side_effect = Exception("Audit failed")

        state = {"error": "original error"}

        # Should not raise, should still return
        result = await node_handle_error(state)
        assert result["error"] == "original error"


class TestExtractEvidenceFromAnswer:
    """Additional test cases for evidence extraction."""

    def test_no_citations_still_includes_top_results(self):
        """Test answer with no citations still includes top 3 results."""
        answer = "The project is complete."
        results = SearchResults(
            query="test",
            results=[SearchResultItem(chunk_id="c1", highlights=["done"], score=1.0)],
            reranker="test",
        )
        evidence = _extract_evidence_from_answer(answer, results)
        # Top 3 results are always included per function logic
        assert len(evidence) >= 1

    def test_none_results(self):
        """Test with None results."""
        answer = "Based on [Source 1]..."
        evidence = _extract_evidence_from_answer(answer, None)
        assert evidence == []

    def test_citation_beyond_results(self):
        """Test citation index beyond available results."""
        answer = "According to [Source 5], this is true."
        results = SearchResults(
            query="test",
            results=[SearchResultItem(chunk_id="c1", highlights=["info"], score=1.0)],
            reranker="test",
        )
        evidence = _extract_evidence_from_answer(answer, results)
        # Should not include evidence for Source 5 since only 1 result exists
        assert len(evidence) <= 1


class TestNodeAssembleContextEdgeCases:
    """Edge case tests for node_assemble_context."""

    def test_empty_results_list(self):
        """Test with empty results list."""
        results = SearchResults(
            query="test",
            results=[],
            reranker="test",
        )
        state = {"retrieval_results": results}
        output = node_assemble_context(state)
        assert output["assembled_context"] == ""

    def test_results_with_empty_highlights(self):
        """Test results with empty highlights."""
        results = SearchResults(
            query="test",
            results=[
                SearchResultItem(chunk_id="c1", highlights=[], score=0.9),
            ],
            reranker="test",
        )
        state = {"retrieval_results": results}
        output = node_assemble_context(state)
        # Should handle empty highlights gracefully
        context = output["assembled_context"]
        assert "[Source 1 (ID: c1)]" in context


class TestExtractPatterns:
    """Test the _extract_patterns helper function."""

    def test_extract_patterns_basic(self):
        """Test basic pattern extraction."""
        import re

        from cortex.orchestration.nodes import _extract_patterns

        pattern = re.compile(r"\b(\w+\.pdf)\b", re.IGNORECASE)
        seen = set()
        mentions = []
        text = "Please review report.pdf and summary.pdf"

        _extract_patterns(text, pattern, seen, mentions)

        assert "report.pdf" in mentions
        assert "summary.pdf" in mentions
        assert len(mentions) == 2

    def test_extract_patterns_deduplication(self):
        """Test that duplicate patterns are not extracted twice."""
        import re

        from cortex.orchestration.nodes import _extract_patterns

        pattern = re.compile(r"\b(\w+\.pdf)\b", re.IGNORECASE)
        seen = set()
        mentions = []
        text = "See report.pdf and then report.pdf again"

        _extract_patterns(text, pattern, seen, mentions)

        assert mentions.count("report.pdf") == 1

    def test_extract_patterns_min_length(self):
        """Test minimum length filtering."""
        import re

        from cortex.orchestration.nodes import _extract_patterns

        pattern = re.compile(r"attached:\s+(\w+)", re.IGNORECASE)
        seen = set()
        mentions = []
        text = "attached: x and attached: document"

        _extract_patterns(text, pattern, seen, mentions, min_len=2)

        assert "document" in mentions
        assert "x" not in mentions


class TestNodeQueryGraph:
    """Test node_query_graph function."""

    def test_node_query_graph_empty_query(self):
        """Test with empty query returns empty context."""
        from cortex.orchestration.nodes import node_query_graph

        state = {"query": "", "tenant_id": "test"}
        result = node_query_graph(state)
        assert result["graph_context"] == ""

    def test_node_query_graph_no_tenant(self):
        """Test with no tenant_id returns empty context."""
        from cortex.orchestration.nodes import node_query_graph

        state = {"query": "test query", "tenant_id": None}
        result = node_query_graph(state)
        assert result["graph_context"] == ""

    @patch("cortex.orchestration.nodes._extract_entity_mentions")
    def test_node_query_graph_no_mentions(self, mock_extract):
        """Test with no entity mentions returns empty context."""
        from cortex.orchestration.nodes import node_query_graph

        mock_extract.return_value = []
        state = {"query": "test query", "tenant_id": "test"}
        result = node_query_graph(state)
        assert result["graph_context"] == ""


class TestNodeRetrieveContext:
    """Test node_retrieve_context function."""

    @patch("cortex.orchestration.nodes.tool_kb_search_hybrid", new_callable=AsyncMock)
    async def test_node_retrieve_context_success(self, mock_search):
        """Test successful retrieval."""
        from cortex.orchestration.nodes import node_retrieve_context

        mock_results = SearchResults(
            query="test",
            results=[SearchResultItem(chunk_id="c1", highlights=["data"], score=0.9)],
            reranker="test",
        )
        mock_search.return_value = mock_results

        state = {
            "query": "test query",
            "classification": None,
            "tenant_id": "test",
            "user_id": "user1",
        }
        result = await node_retrieve_context(state)
        assert "retrieval_results" in result

    @patch("cortex.orchestration.nodes.tool_kb_search_hybrid", new_callable=AsyncMock)
    async def test_node_retrieve_context_error(self, mock_search):
        """Test retrieval error handling."""
        from cortex.orchestration.nodes import node_retrieve_context

        mock_search.side_effect = Exception("Search failed")

        state = {
            "query": "test query",
            "classification": None,
            "tenant_id": "test",
            "user_id": "user1",
        }
        result = await node_retrieve_context(state)
        assert "error" in result


class TestNodeGenerateAnswerEdgeCases:
    """Additional tests for node_generate_answer."""

    def test_node_generate_answer_empty_context(self):
        """Test answer generation with no context."""
        state = {
            "query": "test question",
            "assembled_context": "",
            "graph_context": "",
            "retrieval_results": None,
        }

        output = node_generate_answer(state)
        answer = output["answer"]
        assert "could not find" in answer.answer_markdown.lower()
        assert answer.confidence_overall == 0.0

    @patch("cortex.orchestration.nodes.complete_text")
    def test_node_generate_answer_with_graph_context(self, mock_complete):
        """Test answer generation with graph context only."""
        mock_complete.return_value = "The answer based on graph."

        state = {
            "query": "test",
            "assembled_context": "",
            "graph_context": "Entity: John (Person) - CEO",
            "retrieval_results": None,
        }

        output = node_generate_answer(state)
        answer = output["answer"]
        assert answer.answer_markdown == "The answer based on graph."
