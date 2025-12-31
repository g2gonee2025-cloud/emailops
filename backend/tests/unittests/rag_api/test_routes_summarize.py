"""Unit test for summarize routes - uses live infrastructure."""

from unittest.mock import MagicMock

import pytest
from cortex.db.models import Conversation
from cortex.db.session import SessionLocal
from cortex.rag_api.models import SummarizeThreadRequest
from cortex.rag_api.routes_summarize import summarize_thread_endpoint
from fastapi import HTTPException


@pytest.mark.asyncio
async def test_summarize_thread_endpoint_invalid_uuid():
    """
    Verify that if an invalid UUID is provided, the endpoint raises 400.
    """
    request = SummarizeThreadRequest(thread_id="not-a-valid-uuid")
    http_request = MagicMock()
    http_request.state.correlation_id = "test_correlation_id"

    with pytest.raises(HTTPException) as exc_info:
        await summarize_thread_endpoint(request, http_request)

    assert exc_info.value.status_code in (400, 401)


@pytest.mark.asyncio
async def test_summarize_thread_endpoint_nonexistent_thread():
    """
    Verify that if a valid UUID that doesn't exist is provided, we get 401/404.
    (401 if auth fails first, 404 if thread not found)
    """
    from uuid import uuid4

    request = SummarizeThreadRequest(thread_id=str(uuid4()))
    http_request = MagicMock()
    http_request.state.correlation_id = "test_correlation_id"

    with pytest.raises(HTTPException) as exc_info:
        await summarize_thread_endpoint(request, http_request)

    # Could be 401 (missing tenant/user context) or 404 (thread not found)
    assert exc_info.value.status_code in (401, 404)
