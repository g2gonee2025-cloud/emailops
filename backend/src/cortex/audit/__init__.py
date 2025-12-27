"""
Audit Logging Module.

Implements §11.4 of the Canonical Blueprint.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional
from uuid import uuid4

from cortex.db.models import AuditLog
from cortex.db.session import AsyncSession, get_async_db_session
from cortex.observability import trace_operation
from cortex.safety.policy_enforcer import PolicyDecision
from pydantic import BaseModel, Field
from sqlalchemy import select

logger = logging.getLogger(__name__)


# AuditEntry model for structured audit data
class AuditEntry(BaseModel):
    """Structured audit entry for API responses."""

    tenant_id: str
    user_or_agent: str
    action: str
    # PII Risk: Snapshots are for in-memory use and hashing; not for direct logging.
    input_snapshot: Dict[str, Any]
    output_snapshot: Optional[Dict[str, Any]] = None
    policy_decision: Optional[PolicyDecision] = None
    ts: datetime
    correlation_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AuditConfirmation(BaseModel):
    """Confirmation receipt for an audit event."""

    audit_id: str
    input_hash: Optional[str] = None
    output_hash: Optional[str] = None
    status: Literal["recorded", "failed"]


async def log_audit_event(
    tenant_id: str,
    user_or_agent: str,
    action: str,
    input_data: Optional[Any] = None,
    output_data: Optional[Any] = None,
    input_hash: Optional[str] = None,
    output_hash: Optional[str] = None,
    policy_decisions: Optional[Dict[str, Any]] = None,
    risk_level: Literal["low", "medium", "high"] = "low",
    correlation_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    db_session: Optional[AsyncSession] = None,
) -> str:
    """
    Record an audit event to the database asynchronously.

    Blueprint §11.4:
    * ts (UTC)
    * tenant_id
    * user_or_agent
    * action
    * input_hash / output_hash
    * policy_decisions
    * risk_level
    * correlation_id
    """
    record_id = str(uuid4())

    async def _write_audit_log(session: AsyncSession):
        session.add(record)
        # The context manager will handle commit/rollback

    try:
        if input_data and not input_hash:
            try:
                input_str = json.dumps(input_data, sort_keys=True, default=str)
                input_hash = hashlib.sha256(input_str.encode("utf-8")).hexdigest()
            except TypeError:
                input_hash = "serialization_failed"

        if output_data and not output_hash:
            try:
                output_str = json.dumps(output_data, sort_keys=True, default=str)
                output_hash = hashlib.sha256(output_str.encode("utf-8")).hexdigest()
            except TypeError:
                output_hash = "serialization_failed"

        final_metadata = dict(metadata or {})
        if correlation_id:
            final_metadata["correlation_id"] = correlation_id

        record = AuditLog(
            audit_id=record_id,
            ts=datetime.now(timezone.utc),
            tenant_id=tenant_id,
            user_or_agent=user_or_agent,
            action=action,
            input_hash=input_hash,
            output_hash=output_hash,
            policy_decisions=policy_decisions,
            risk_level=risk_level,
            metadata_=final_metadata,
        )

        if db_session:
            await _write_audit_log(db_session)
        else:
            async with get_async_db_session(tenant_id=tenant_id) as session:
                await _write_audit_log(session)

        return record_id

    except Exception as e:
        # Security: Log a sanitized message. Avoid logging 'e' or exc_info=True
        # to prevent leaking sensitive data from snapshots on serialization failure.
        logger.critical("AUDIT LOGGING FAILED: Could not persist audit record.")
        raise


@trace_operation("tool_audit_log")
async def tool_audit_log(
    tenant_id: str,
    user_or_agent: str,
    action: str,
    input_snapshot: Dict[str, Any],
    output_snapshot: Optional[Dict[str, Any]] = None,
    policy_decision: Optional[PolicyDecision] = None,
    correlation_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    db_session: Optional[AsyncSession] = None,
) -> AuditConfirmation:
    """
    Create and persist an audit log entry. Returns a confirmation, not raw data.

    Blueprint §11.4 - tool_audit_log:
    * Creates AuditEntry with all required fields
    * Writes to audit_log table
    * Returns the created entry for confirmation

    Args:
        tenant_id: Tenant ID for multi-tenancy
        user_or_agent: User ID or agent name
        action: Action being audited (e.g., "draft_email", "search")
        input_snapshot: Snapshot of input data (PII)
        output_snapshot: Snapshot of output data (optional PII)
        policy_decision: Policy decision if applicable
        correlation_id: Request correlation ID for tracing
        metadata: Additional metadata

    Returns:
        AuditConfirmation with hashes and status.
    """
    ts = datetime.now(timezone.utc)
    risk_level: Literal["low", "medium", "high"] = "low"
    if policy_decision:
        risk_level = policy_decision.risk_level

    try:
        input_hash = hashlib.sha256(
            json.dumps(input_snapshot, sort_keys=True, default=str).encode()
        ).hexdigest()

        output_hash = None
        if output_snapshot:
            output_hash = hashlib.sha256(
                json.dumps(output_snapshot, sort_keys=True, default=str).encode()
            ).hexdigest()

        audit_id = await log_audit_event(
            tenant_id=tenant_id,
            user_or_agent=user_or_agent,
            action=action,
            input_hash=input_hash,
            output_hash=output_hash,
            policy_decisions=policy_decision.model_dump() if policy_decision else None,
            risk_level=risk_level,
            correlation_id=correlation_id,
            metadata={
                **(metadata or {}),
                "input_keys": list(input_snapshot.keys()),
                "output_keys": list(output_snapshot.keys()) if output_snapshot else [],
            },
            db_session=db_session,
        )
        return AuditConfirmation(
            audit_id=audit_id,
            input_hash=input_hash,
            output_hash=output_hash,
            status="recorded",
        )
    except Exception:
        logger.error("Failed to create audit log entry for action: %s", action)
        return AuditConfirmation(audit_id=str(uuid4()), status="failed")


async def get_audit_trail(
    tenant_id: str,
    action: Optional[str] = None,
    user_or_agent: Optional[str] = None,
    correlation_id: Optional[str] = None,
    since: Optional[datetime] = None,
    limit: int = 100,
) -> list[AuditLog]:
    """
    Query audit trail with filters asynchronously.

    Args:
        tenant_id: Required tenant filter
        action: Optional action filter
        user_or_agent: Optional user/agent filter
        correlation_id: Optional correlation ID filter
        since: Optional timestamp filter (events after this time)
        limit: Maximum number of results (max 1000)

    Returns:
        List of matching AuditLog records
    """
    # Security: Enforce a hard limit to prevent resource exhaustion.
    limit = min(limit, 1000)

    try:
        async with get_async_db_session(tenant_id=tenant_id) as session:
            query = select(AuditLog)

            if action:
                query = query.where(AuditLog.action == action)
            if user_or_agent:
                query = query.where(AuditLog.user_or_agent == user_or_agent)
            if correlation_id:
                query = query.where(
                    AuditLog.metadata_["correlation_id"].astext == correlation_id
                )
            if since:
                query = query.where(AuditLog.ts >= since)

            query = query.order_by(AuditLog.ts.desc()).limit(limit)

            result = await session.execute(query)
            return result.scalars().all()

    except Exception:
        logger.error("Failed to query audit trail", exc_info=False)
        return []


