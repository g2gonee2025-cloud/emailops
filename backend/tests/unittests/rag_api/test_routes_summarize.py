from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from cortex.rag_api.models import SummarizeThreadRequest
from cortex.rag_api.routes_summarize import summarize_thread_endpoint


@pytest.mark.asyncio
async def test_summarize_thread_endpoint_service_error():
    """
    Verify that if the summarize service raises an exception, the endpoint
    raises a 500 HTTPException.
    """
    # Arrange
    request = SummarizeThreadRequest(thread_id="test_thread")
    http_request = MagicMock()
    http_request.state.correlation_id = "test_correlation_id"
    mock_user = "test_user"

    # Patch the service to simulate a failure
    with patch(
        "cortex.rag_api.routes_summarize.summarize_thread_service",
        new_callable=AsyncMock,
        side_effect=ValueError("Service layer error"),
    ) as mock_summarize_service:
        # Act & Assert
        with pytest.raises(HTTPException) as exc_info:
            await summarize_thread_endpoint(request, http_request, user=mock_user)

        # Assert that the correct exception is raised by the endpoint
        assert exc_info.value.status_code == 400
        assert "Service layer error" in exc_info.value.detail

        # Ensure the service was called
        mock_summarize_service.assert_awaited_once()
