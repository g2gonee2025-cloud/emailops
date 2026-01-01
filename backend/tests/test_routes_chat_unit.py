"""
Live integration tests for routes_chat using real LLM via CPU fallback.
"""

from unittest.mock import MagicMock

import pytest
from cortex.rag_api.routes_chat import (
    ChatMessage,
    ChatRequest,
    _decide_action,
    chat_endpoint,
)


class TestRoutesChat:
    """Test chat routes with live LLM calls."""

    @pytest.fixture
    def mock_request(self):
        """Create mock HTTP request with correlation ID."""
        req = MagicMock()
        req.state.correlation_id = "test-corr-id"
        return req

    @pytest.mark.asyncio
    async def test_chat_endpoint_simple_greeting(self, mock_request):
        """Test chat endpoint with a simple greeting - live LLM call."""
        chat_req = ChatRequest(
            messages=[ChatMessage(role="user", content="Hello, how are you?")],
            debug=True,
        )

        resp = await chat_endpoint(chat_req, mock_request)

        # Verify response structure
        assert resp is not None
        assert hasattr(resp, "action")
        assert hasattr(resp, "reply")
        assert resp.correlation_id == "test-corr-id"

    @pytest.mark.asyncio
    async def test_chat_endpoint_search_query(self, mock_request):
        """Test chat endpoint with search-like query - live LLM call."""
        chat_req = ChatRequest(
            messages=[
                ChatMessage(role="user", content="Find emails about insurance claims")
            ],
            debug=True,
        )

        resp = await chat_endpoint(chat_req, mock_request)

        # Should return valid response regardless of action chosen
        assert resp is not None
        assert resp.action in ("answer", "search", "summarize", "draft")

    @pytest.mark.asyncio
    async def test_decide_action_with_live_llm(self):
        """Test action decision with live LLM."""
        req = ChatRequest(
            messages=[ChatMessage(role="user", content="What is the weather today?")]
        )

        decision = await _decide_action(req, req.messages, "What is the weather today?")

        # Should return valid decision
        assert decision is not None
        assert hasattr(decision, "action")
        assert hasattr(decision, "reason")
        assert decision.action in ("answer", "search", "summarize", "draft", "clarify")

    @pytest.mark.asyncio
    async def test_chat_with_context(self, mock_request):
        """Test chat with conversation context - live LLM call."""
        chat_req = ChatRequest(
            messages=[
                ChatMessage(role="user", content="Tell me about flood claims"),
                ChatMessage(role="assistant", content="I can help with that."),
                ChatMessage(role="user", content="How many are pending?"),
            ],
            debug=True,
        )

        resp = await chat_endpoint(chat_req, mock_request)

        assert resp is not None
        assert hasattr(resp, "action")
