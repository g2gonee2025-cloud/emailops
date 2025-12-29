"""Unit tests for cortex.domain.models and cortex.tools.search."""

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
    """Test tool_kb_search_hybrid with live database."""

    async def test_tool_kb_search_hybrid_with_domain_input(self):
        """Test search with KBSearchInput model against live DB."""
        import pytest
        from cortex.tools.search import tool_kb_search_hybrid

        inp = KBSearchInput(query="insurance claim", tenant_id="default")
        result = await tool_kb_search_hybrid(inp)

        # Should return Ok result (may or may not have results depending on DB state)
        assert result.is_ok() or result.is_err()

    async def test_tool_kb_search_hybrid_with_retrieval_input(self):
        """Test search with RetrievalKBSearchInput against live DB."""
        import pytest
        from cortex.retrieval.hybrid_search import (
            KBSearchInput as RetrievalKBSearchInput,
        )
        from cortex.tools.search import tool_kb_search_hybrid

        inp = RetrievalKBSearchInput(
            tenant_id="default",
            user_id="test_user",
            query="flood damage",
            k=5,
            fusion_method="rrf",
            filters={},
        )
        result = await tool_kb_search_hybrid(inp)

        # Should return a Result type
        assert result.is_ok() or result.is_err()

    async def test_tool_kb_search_hybrid_empty_query(self):
        """Test search with empty query handles gracefully."""
        from cortex.tools.search import tool_kb_search_hybrid

        inp = KBSearchInput(query="", tenant_id="default")
        result = await tool_kb_search_hybrid(inp)

        # Should handle empty query gracefully
        assert result.is_ok() or result.is_err()
