"""
Draft API Routes.

Implements §9.2 of the Canonical Blueprint.
"""

from __future__ import annotations

import logging

from cortex.common.exceptions import CortexError
from cortex.observability import trace_operation
from cortex.rag_api.models import DraftEmailRequest, DraftEmailResponse
from cortex.rag_api.services import draft_email_service
from cortex.security.dependencies import get_current_user
from fastapi import APIRouter, Depends, HTTPException, Request

logger = logging.getLogger(__name__)
router = APIRouter()


def get_draft_graph(http_request: Request):
    """
    Get pre-compiled graph from app.state.

    This is intended to be used as a FastAPI dependency.
    """
    try:
        return http_request.app.state.graphs["draft"]
    except (AttributeError, KeyError) as e:
        logger.exception("Draft graph not found in app.state.graphs. Check lifespan configuration.")
        raise HTTPException(
            status_code=500,
            detail="Internal server error: Draft service not configured",
        ) from e


@router.post(
    "/draft-email",
    response_model=DraftEmailResponse,
    dependencies=[Depends(get_current_user)],
)
@trace_operation("api_draft_email")
async def draft_endpoint(
    request: DraftEmailRequest,
    http_request: Request,
    graph=Depends(get_draft_graph),
) -> DraftEmailResponse:
    """
    Draft email endpoint.

    Blueprint §9.2:
    * POST /api/v1/draft-email
    * Request: DraftEmailRequest
    * Response: DraftEmailResponse with correlation_id
    """
    correlation_id = getattr(http_request.state, "correlation_id", None)

    try:
        result = await draft_email_service(
            graph=graph,
            request=request,
            correlation_id=correlation_id,
        )

        return DraftEmailResponse(
            correlation_id=correlation_id,
            draft=result["draft"],
            iterations=result["iterations"],
        )

    except CortexError as e:
        # Handle domain-specific exceptions
        raise HTTPException(status_code=400, detail=e.to_dict())
    except HTTPException:
        # Re-raise known HTTP exceptions
        raise
    except Exception as e:
        # Security Hardening: Log the full error but return a generic message
        logger.exception("Draft failed due to an unexpected error")
        raise HTTPException(status_code=500, detail="An internal error occurred while drafting the email.")
