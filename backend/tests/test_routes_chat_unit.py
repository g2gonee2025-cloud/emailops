from unittest.mock import MagicMock, patch

import pytest
from cortex.rag_api.routes_chat import (
    ChatActionDecision,
    ChatMessage,
    ChatRequest,
    ChatResponse,
    _decide_action,
    chat_endpoint,
)


class TestRoutesChat:
    @pytest.fixture
    def mock_request(self):
        req = MagicMock()
        req.state.correlation_id = "test-corr-id"
        return req

    @pytest.mark.asyncio
    @patch("cortex.rag_api.routes_chat._decide_action")
    @patch("cortex.rag_api.routes_chat._handle_answer")
    @patch("cortex.rag_api.routes_chat._log_chat_audit")
    async def test_chat_endpoint_answer_flow(
        self, mock_audit, mock_handle, mock_decide, mock_request
    ):
        # Setup
        chat_req = ChatRequest(
            messages=[ChatMessage(role="user", content="Hello")], debug=True
        )
        mock_decide.return_value = ChatActionDecision(
            action="answer", reason="Just answering"
        )

        mock_response = ChatResponse(
            correlation_id="test-corr-id",
            action="answer",
            reply="Hello back",
            answer=None,
        )
        mock_handle.return_value = mock_response

        # Execute
        resp = await chat_endpoint(chat_req, mock_request)

        # Verify
        assert resp.action == "answer"
        assert resp.reply == "Hello back"
        mock_decide.assert_called_once()
        mock_handle.assert_called_once()
        mock_audit.assert_called_once()

    @pytest.mark.asyncio
    @patch("cortex.rag_api.routes_chat._decide_action")
    @patch("cortex.rag_api.routes_chat._handle_search")
    async def test_chat_endpoint_search_flow(
        self, mock_handle, mock_decide, mock_request
    ):
        chat_req = ChatRequest(
            messages=[ChatMessage(role="user", content="Search this")], debug=True
        )
        mock_decide.return_value = ChatActionDecision(
            action="search", reason="Searching"
        )

        mock_response = ChatResponse(
            correlation_id="test-corr-id", action="search", reply="Results found"
        )
        mock_handle.return_value = mock_response

        resp = await chat_endpoint(chat_req, mock_request)
        assert resp.action == "search"

    @pytest.mark.asyncio
    @patch("cortex.rag_api.routes_chat._decide_action")
    @patch("cortex.rag_api.routes_chat._handle_summarize")
    async def test_chat_endpoint_summarize_flow(
        self, mock_handle, mock_decide, mock_request
    ):
        chat_req = ChatRequest(
            messages=[ChatMessage(role="user", content="Summarize this")], debug=True
        )
        mock_decide.return_value = ChatActionDecision(
            action="summarize", reason="Summarizing"
        )

        mock_response = ChatResponse(
            correlation_id="test-corr-id", action="summarize", reply="Summary here"
        )
        mock_handle.return_value = mock_response

        resp = await chat_endpoint(chat_req, mock_request)
        assert resp.action == "summarize"

    def test_decide_action_call(self):
        # Mocking complete_json to verify _decide_action logic without calling LLM
        with patch("cortex.rag_api.routes_chat.complete_json") as mock_complete:
            mock_complete.return_value = {"action": "answer", "reason": "test"}

            req = ChatRequest(messages=[ChatMessage(role="user", content="hi")])
            decision = _decide_action(req, req.messages, "hi")

            assert decision.action == "answer"
            assert decision.reason == "test"
            mock_complete.assert_called_once()
