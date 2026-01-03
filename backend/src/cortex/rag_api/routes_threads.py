"""
Thread/Conversation API Routes.

Provides endpoints for listing and searching conversations.
"""

from __future__ import annotations

import logging
from typing import Any

from cortex.context import tenant_id_ctx
from cortex.db.models import Conversation
from cortex.db.session import SessionLocal, set_session_tenant
from cortex.security.auth import get_current_user
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import func, or_

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/threads", tags=["threads"])


class ThreadListItem(BaseModel):
    """Thread/conversation list item for dropdown selection."""

    conversation_id: str
    subject: str | None = None
    smart_subject: str | None = None
    folder_name: str
    participants_preview: str | None = None
    latest_date: str | None = None


class ThreadListResponse(BaseModel):
    """Response for thread listing endpoint."""

    threads: list[ThreadListItem] = Field(default_factory=list)
    total_count: int = 0
    has_more: bool = False


def _require_tenant_id() -> str:
    tenant_id = tenant_id_ctx.get()
    if not isinstance(tenant_id, str) or not tenant_id.strip():
        raise HTTPException(status_code=401, detail="Tenant context missing")
    return tenant_id


def _extract_participants_preview(participants: Any, max_names: int = 3) -> str | None:
    """Extract a preview string from participants JSONB."""
    if not participants or not isinstance(participants, list):
        return None

    names = []
    for p in participants[:max_names]:
        if isinstance(p, dict):
            name = p.get("name") or p.get("smtp") or p.get("email")
            if name:
                names.append(str(name))

    if not names:
        return None

    preview = ", ".join(names)
    if len(participants) > max_names:
        preview += f" +{len(participants) - max_names} more"

    return preview


@router.get(
    "",
    response_model=ThreadListResponse,
    dependencies=[Depends(get_current_user)],
)
async def list_threads(
    q: str | None = Query(default=None, description="Search query for subject/folder"),
    limit: int = Query(default=50, ge=1, le=200, description="Max threads to return"),
    offset: int = Query(default=0, ge=0, description="Offset for pagination"),
):
    """
    List conversations/threads with optional search.

    Returns a list of threads for dropdown selection in draft/ask views.
    Searches across subject, smart_subject, and folder_name fields.
    """
    tenant_id = _require_tenant_id()

    try:
        with SessionLocal() as session:
            set_session_tenant(session, tenant_id)

            query = session.query(Conversation).filter(
                Conversation.tenant_id == tenant_id
            )

            if q and q.strip():
                search_term = f"%{q.strip()}%"
                query = query.filter(
                    or_(
                        Conversation.subject.ilike(search_term),
                        Conversation.smart_subject.ilike(search_term),
                        Conversation.folder_name.ilike(search_term),
                    )
                )

            total_count = query.count()

            conversations = (
                query.order_by(Conversation.latest_date.desc().nullslast())
                .offset(offset)
                .limit(limit + 1)
                .all()
            )

            has_more = len(conversations) > limit
            conversations = conversations[:limit]

            threads = []
            for conv in conversations:
                threads.append(
                    ThreadListItem(
                        conversation_id=str(conv.conversation_id),
                        subject=conv.subject,
                        smart_subject=conv.smart_subject,
                        folder_name=conv.folder_name,
                        participants_preview=_extract_participants_preview(
                            conv.participants
                        ),
                        latest_date=(
                            conv.latest_date.isoformat() if conv.latest_date else None
                        ),
                    )
                )

            return ThreadListResponse(
                threads=threads,
                total_count=total_count,
                has_more=has_more,
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to list threads: %s", e)
        raise HTTPException(status_code=500, detail="Failed to list threads")
