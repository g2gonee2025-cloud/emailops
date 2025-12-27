
from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from cortex.audit import get_audit_trail, log_audit_event, tool_audit_log


@pytest.mark.asyncio
async def test_log_audit_event_writes_to_db(async_db_session: AsyncSession):
    """Verify that log_audit_event correctly writes a record to the database."""
    tenant_id = "audit-test-tenant"
    user = "test-user"
    action = "test-action"
    input_data = {"key": "value"}

    await log_audit_event(
        tenant_id=tenant_id,
        user_or_agent=user,
        action=action,
        input_data=input_data,
        db_session=async_db_session,
    )

    # Use get_audit_trail to verify the write
    trail = await get_audit_trail(tenant_id=tenant_id, action=action, limit=1)
    assert len(trail) == 1
    assert trail[0].user_or_agent == user
    assert trail[0].action == action


@pytest.mark.asyncio
async def test_tool_audit_log_returns_confirmation(async_db_session: AsyncSession):
    """Check that tool_audit_log returns a sanitized confirmation, not raw data."""
    tenant_id = "audit-tool-test-tenant"
    user = "test-tool-user"
    action = "test-tool-action"
    input_snapshot = {"secret": "data"}

    confirmation = await tool_audit_log(
        tenant_id=tenant_id,
        user_or_agent=user,
        action=action,
        input_snapshot=input_snapshot,
        db_session=async_db_session,
    )

    assert confirmation.status == "recorded"
    assert confirmation.input_hash is not None
    assert not hasattr(confirmation, "input_snapshot")

    # Verify the data was actually written
    trail = await get_audit_trail(tenant_id=tenant_id, action=action, limit=1)
    assert len(trail) == 1
    assert trail[0].input_hash == confirmation.input_hash
