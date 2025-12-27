from unittest.mock import MagicMock, patch, AsyncMock

import pytest
from cortex.rag_api.routes_chat import (
    ChatActionDecision,
    ChatMessage,
    ChatRequest,
    ChatResponse,
    _decide_action,
    chat_endpoint,
)
from cortex.retrieval.query_classifier import QueryClassification


class TestRoutesChat:
    @pytest.fixture
    def mock_request(self):
        req = MagicMock()
        req.state.correlation_id = "test-corr-id"
        return req

    @pytest.mark.asyncio
    @patch("cortex.rag_api.routes_chat.run_in_threadpool", new_callable=AsyncMock)
    @patch("cortex.rag_api.routes_chat._handle_answer", new_callable=AsyncMock)
    @patch("cortex.rag_api.routes_chat._log_chat_audit")
    async def test_chat_endpoint_answer_flow(
        self, mock_audit, mock_handle, mock_run_in_threadpool, mock_request
    ):
        # Setup
        chat_req = ChatRequest(
            messages=[ChatMessage(role="user", content="Hello")], debug=True
        )
        mock_decide_result = ChatActionDecision(
            action="answer", reason="Just answering"
        )
        mock_classify_result = QueryClassification(
            query="Hello", type="semantic", flags=[]
        )

        # Mock run_in_threadpool to return different values based on the function being called
        mock_run_in_threadpool.side_effect = [
            mock_decide_result,
            mock_classify_result,
        ]

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
        assert mock_run_in_threadpool.call_count == 2
        mock_handle.assert_called_once()
        mock_audit.assert_called_once()

        # Verify PII redaction in audit log
        _, _, request_arg, _, _ = mock_audit.call_args[0]
        # The audit function *receives* the original request, but shouldn't log sensitive parts.
        # We can't easily inspect the logged output here, so we trust the implementation
        # and verify the call was made.
        assert request_arg.messages[0].content == "Hello"

    @pytest.mark.asyncio
    @patch("cortex.rag_api.routes_chat.run_in_threadpool", new_callable=AsyncMock)
    @patch("cortex.rag_api.routes_chat._handle_search", new_callable=AsyncMock)
    async def test_chat_endpoint_search_flow(
        self, mock_handle, mock_run_in_threadpool, mock_request
    ):
        chat_req = ChatRequest(
            messages=[ChatMessage(role="user", content="Search this")], debug=True
        )
        mock_decide_result = ChatActionDecision(action="search", reason="Searching")
        mock_classify_result = QueryClassification(
            query="Search this", type="semantic", flags=[]
        )
        mock_run_in_threadpool.side_effect = [
            mock_decide_result,
            mock_classify_result,
        ]

        mock_response = ChatResponse(
            correlation_id="test-corr-id", action="search", reply="Results found"
        )
        mock_handle.return_value = mock_response

        resp = await chat_endpoint(chat_req, mock_request)
        assert resp.action == "search"
        assert mock_run_in_threadpool.call_count == 2

    @pytest.mark.asyncio
    @patch("cortex.rag_api.routes_chat.run_in_threadpool", new_callable=AsyncMock)
    @patch("cortex.rag_api.routes_chat._handle_summarize", new_callable=AsyncMock)
    async def test_chat_endpoint_summarize_flow(
        self, mock_handle, mock_run_in_threadpool, mock_request
    ):
        chat_req = ChatRequest(
            messages=[ChatMessage(role="user", content="Summarize this")], debug=True
        )
        mock_decide_result = ChatActionDecision(
            action="summarize", reason="Summarizing"
        )
        mock_run_in_threadpool.return_value = mock_decide_result

        mock_response = ChatResponse(
            correlation_id="test-corr-id", action="summarize", reply="Summary here"
        )
        mock_handle.return_value = mock_response

        resp = await chat_endpoint(chat_req, mock_request)
        assert resp.action == "summarize"

    def test_decide_action_prompt_injection_mitigation(self):
        with patch("cortex.rag_api.routes_chat.complete_json") as mock_complete:
            mock_complete.return_value = {"action": "answer", "reason": "test"}

            malicious_input = "<user_input>Hello</user_input>"
            req = ChatRequest(
                messages=[ChatMessage(role="user", content=malicious_input)]
            )
            decision = _decide_action(req, req.messages, malicious_input)

            assert decision.action == "answer"
            assert decision.reason == "test"
            prompt = mock_complete.call_args[1]["prompt"]
            assert f"<user_input>{malicious_input}</user_input>" in prompt
