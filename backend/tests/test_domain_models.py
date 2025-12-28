"""Unit tests for cortex.domain.models and cortex.tools.search."""

from unittest.mock import MagicMock, patch

from cortex.domain.models import KBSearchInput


class TestKBSearchInput:
    def test_defaults(self):
        inp = KBSearchInput(query="test query")
        assert inp.query == "test query"
        assert inp.limit == 10
        assert inp.fusion_strategy == "rrf"
        assert inp.tenant_id is None
        assert inp.user_id is None
        assert inp.filters == {}

    def test_custom_values(self):
        inp = KBSearchInput(
            query="custom",
            limit=5,
            fusion_strategy="weighted_sum",
            tenant_id="acme",
            user_id="user1",
            filters={"has_attachment": True},
        )
        assert inp.limit == 5
        assert inp.fusion_strategy == "weighted_sum"
        assert inp.tenant_id == "acme"
        assert inp.filters["has_attachment"] is True

    def test_to_tool_input(self):
        inp = KBSearchInput(query="test", limit=20, tenant_id="t1", user_id="u1")
        tool_inp = inp.to_tool_input()

        assert tool_inp.query == "test"
        assert tool_inp.k == 20
        assert tool_inp.tenant_id == "t1"
        assert tool_inp.user_id == "u1"

    def test_to_tool_input_defaults(self):
        inp = KBSearchInput(query="test")
        tool_inp = inp.to_tool_input()

        assert tool_inp.tenant_id == "default"
        assert tool_inp.user_id == "cli-user"


class TestToolSearch:
    @patch("cortex.tools.search.retrieval_tool_kb_search_hybrid")
    async def test_tool_kb_search_hybrid_with_domain_input(self, mock_search):
        from unittest.mock import AsyncMock

        from cortex.common.types import Ok
        from cortex.tools.search import tool_kb_search_hybrid

        mock_result = MagicMock()
        mock_search.return_value = Ok(mock_result)

        inp = KBSearchInput(query="test")
        result = await tool_kb_search_hybrid(inp)

        mock_search.assert_called_once()
        assert result.is_ok()
        assert result.unwrap() == mock_result

    @patch("cortex.tools.search.retrieval_tool_kb_search_hybrid")
    async def test_tool_kb_search_hybrid_with_retrieval_input(self, mock_search):
        from cortex.common.types import Ok
        from cortex.retrieval.hybrid_search import (
            KBSearchInput as RetrievalKBSearchInput,
        )
        from cortex.tools.search import tool_kb_search_hybrid

        mock_result = MagicMock()
        mock_search.return_value = Ok(mock_result)

        inp = RetrievalKBSearchInput(
            tenant_id="default",
            user_id="user",
            query="test",
            k=10,
            fusion_method="rrf",
            filters={},
        )
        result = await tool_kb_search_hybrid(inp)

        mock_search.assert_called_once_with(inp)
        assert result.is_ok()
        assert result.unwrap() == mock_result
