from unittest.mock import MagicMock, patch

import pytest
from cortex.rag_api.models import DraftEmailRequest, DraftEmailResponse
from cortex.rag_api.routes_draft import draft_endpoint


class TestRoutesDraft:
    @pytest.fixture
    def mock_request(self):
        req = MagicMock()
        req.state.correlation_id = "test-corr-id"
        return req

    @pytest.mark.asyncio
    @patch("cortex.rag_api.routes_draft.draft_email_service")
    @patch("cortex.rag_api.routes_draft.get_current_user")
    async def test_draft_endpoint(
        self, mock_get_current_user, mock_draft_email_service, mock_request
    ):
        # Setup
        draft_req = DraftEmailRequest(
            instruction="Test instruction",
        )

        # The service returns an EmailDraft object, so we mock that
        mock_draft = {
            "to": ["test@example.com"],
            "cc": [],
            "subject": "Test Subject",
            "body_markdown": "Test body",
        }

        mock_draft_email_service.return_value = {
            "draft": mock_draft,
            "iterations": 1,
        }
        mock_graph = MagicMock()

        # Execute
        resp = await draft_endpoint(draft_req, mock_request, mock_graph)

        # Verify
        assert isinstance(resp, DraftEmailResponse)
        assert resp.correlation_id == "test-corr-id"
        assert resp.draft.to == ["test@example.com"]
        assert resp.draft.subject == "Test Subject"
        assert resp.iterations == 1
        mock_draft_email_service.assert_called_once_with(
            graph=mock_graph,
            request=draft_req,
            correlation_id="test-corr-id",
        )
