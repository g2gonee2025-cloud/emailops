from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from cortex.rag_api.models import SummarizeThreadRequest
from cortex.rag_api.routes_summarize import summarize_thread_endpoint
from fastapi import HTTPException


@pytest.mark.asyncio
async def test_summarize_thread_endpoint_graph_error():
    """
    Verify that if the summarization graph returns an error, the endpoint
    raises a 500 HTTPException with a specific error message.
    """
    # Arrange
    request = SummarizeThreadRequest(thread_id="test_thread")
    http_request = MagicMock()
    http_request.state.correlation_id = "test_correlation_id"

    # Mock the graph to simulate an error during invocation
    mock_graph = MagicMock()
    mock_graph.ainvoke = AsyncMock(return_value={"error": "Test graph error"})

    # Create mocks for context variables
    mock_tenant_ctx = MagicMock()
    mock_tenant_ctx.get.return_value = "test_tenant"
    mock_user_ctx = MagicMock()
    mock_user_ctx.get.return_value = "test_user"

    # Patch dependencies: context variables and the graph getter
    with (
        patch("cortex.rag_api.routes_summarize.tenant_id_ctx", new=mock_tenant_ctx),
        patch("cortex.rag_api.routes_summarize.user_id_ctx", new=mock_user_ctx),
        patch(
            "cortex.rag_api.routes_summarize.get_summarize_graph",
            new_callable=AsyncMock,
            return_value=mock_graph,
        ),
        patch("cortex.rag_api.routes_summarize.log_audit_event") as mock_audit,
    ):

        # Act & Assert
        with pytest.raises(HTTPException) as exc_info:
            await summarize_thread_endpoint(request, http_request)

        # Assert correct exception is raised
        assert exc_info.value.status_code == 500
        assert exc_info.value.detail == "Summarization workflow failed"

        # Assert that audit logging was NOT called on failure
        mock_audit.assert_not_called()
