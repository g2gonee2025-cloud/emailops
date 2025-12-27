"""
Audit Logging Module.

Implements §11.4 of the Canonical Blueprint.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import UTC, datetime, timezone
from typing import Any, Literal, Optional
from uuid import uuid4

from cortex.db.models import AuditLog
from cortex.db.session import SessionLocal
from cortex.observability import trace_operation
from cortex.safety.policy_enforcer import PolicyDecision
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# AuditEntry model for structured audit data
class AuditEntry(BaseModel):
    """Structured audit entry for API responses."""

    tenant_id: str
    user_or_agent: str
    action: str
    input_snapshot: dict[str, Any]
    output_snapshot: dict[str, Any] | None = None
    policy_decision: PolicyDecision | None = None
    ts: datetime
    correlation_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


def log_audit_event(
    tenant_id: str,
    user_or_agent: str,
    action: str,
    input_data: Any | None = None,
    output_data: Any | None = None,
    input_hash: str | None = None,
    output_hash: str | None = None,
    policy_decisions: dict[str, Any] | None = None,
    risk_level: Literal["low", "medium", "high"] = "low",
    correlation_id: str | None = None,
    metadata: dict[str, Any] | None = None,
    db_session: SessionLocal | None = None,
) -> None:
    """
    Record an audit event to the database.

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

    def _write_audit_log(session):
        from cortex.db.session import set_session_tenant

        set_session_tenant(session, tenant_id)
        session.add(record)
        session.commit()

    try:
        # Calculate hashes if data provided and hash not explicitly provided
        if input_data and not input_hash:
            try:
                input_str = json.dumps(input_data, sort_keys=True, default=str)
                input_hash = hashlib.sha256(input_str.encode("utf-8")).hexdigest()
            except Exception:
                input_hash = "serialization_failed"

        if output_data and not output_hash:
            try:
                output_str = json.dumps(output_data, sort_keys=True, default=str)
                output_hash = hashlib.sha256(output_str.encode("utf-8")).hexdigest()
            except Exception:
                output_hash = "serialization_failed"

        # Include correlation_id in metadata
        final_metadata = dict(metadata or {})
        if correlation_id:
            final_metadata["correlation_id"] = correlation_id

        # Create record
        record = AuditLog(
            audit_id=uuid4(),
            ts=datetime.now(UTC),
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
            _write_audit_log(db_session)
        else:
            # Write to DB (new session to avoid transaction coupling)
            with SessionLocal() as session:
                _write_audit_log(session)

    except Exception as e:
        # Audit logging failure should not crash the app, but must be logged critically
        logger.critical(f"AUDIT LOGGING FAILED: {e}", exc_info=True)


@trace_operation("tool_audit_log")
def tool_audit_log(
    tenant_id: str,
    user_or_agent: str,
    action: str,
    input_snapshot: dict[str, Any],
    output_snapshot: dict[str, Any] | None = None,
    policy_decision: PolicyDecision | None = None,
    correlation_id: str | None = None,
    metadata: dict[str, Any] | None = None,
    db_session: Any | None = None,
) -> AuditEntry:
    """
    Create and persist an audit log entry.

    Blueprint §11.4 - tool_audit_log:
    * Creates AuditEntry with all required fields
    * Writes to audit_log table
    * Returns the created entry for confirmation

    Args:
        tenant_id: Tenant ID for multi-tenancy
        user_or_agent: User ID or agent name
        action: Action being audited (e.g., "draft_email", "search")
        input_snapshot: Snapshot of input data
        output_snapshot: Snapshot of output data (optional)
        policy_decision: Policy decision if applicable
        correlation_id: Request correlation ID for tracing
        metadata: Additional metadata

    Returns:
        AuditEntry with all recorded fields
    """
    ts = datetime.now(UTC)

    # Determine risk level from policy decision or default
    risk_level: Literal["low", "medium", "high"] = "low"
    if policy_decision:
        risk_level = policy_decision.risk_level

    # Create entry model
    entry = AuditEntry(
        tenant_id=tenant_id,
        user_or_agent=user_or_agent,
        action=action,
        input_snapshot=input_snapshot,
        output_snapshot=output_snapshot,
        policy_decision=policy_decision,
        ts=ts,
        correlation_id=correlation_id,
        metadata=metadata or {},
    )

    # Compute hashes
    input_hash = hashlib.sha256(
        json.dumps(input_snapshot, sort_keys=True, default=str).encode()
    ).hexdigest()

    output_hash = None
    if output_snapshot:
        output_hash = hashlib.sha256(
            json.dumps(output_snapshot, sort_keys=True, default=str).encode()
        ).hexdigest()

    # Persist using log_audit_event
    log_audit_event(
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

    return entry


def get_audit_trail(
    tenant_id: str,
    action: str | None = None,
    user_or_agent: str | None = None,
    correlation_id: str | None = None,
    since: datetime | None = None,
    limit: int = 100,
) -> list[AuditLog]:
    """
    Query audit trail with filters.

    Args:
        tenant_id: Required tenant filter
        action: Optional action filter
        user_or_agent: Optional user/agent filter
        correlation_id: Optional correlation ID filter
        since: Optional timestamp filter (events after this time)
        limit: Maximum number of results

    Returns:
        List of matching AuditLog records
    """
    try:
        with SessionLocal() as session:
            query = session.query(AuditLog).filter(AuditLog.tenant_id == tenant_id)

            if action:
                query = query.filter(AuditLog.action == action)
            if user_or_agent:
                query = query.filter(AuditLog.user_or_agent == user_or_agent)
            if correlation_id:
                query = query.filter(
                    AuditLog.metadata_["correlation_id"].astext == correlation_id
                )
            if since:
                query = query.filter(AuditLog.ts >= since)

            query = query.order_by(AuditLog.ts.desc()).limit(limit)

            return query.all()

    except Exception as e:
        logger.error(f"Failed to query audit trail: {e}", exc_info=True)
        return []


def get_audit_log_cli(
    tenant_id: str,
    limit: int = 100,
    since: datetime | None = None,
    user_or_agent: str | None = None,
    action: str | None = None,
    correlation_id: str | None = None,
) -> None:
    """CLI-friendly audit log query."""
    from rich.console import Console
    from rich.table import Table

    console = Console()

    try:
        results = get_audit_trail(
            tenant_id=tenant_id,
            action=action,
            user_or_agent=user_or_agent,
            correlation_id=correlation_id,
            since=since,
            limit=limit,
        )

        if not results:
            console.print("No audit events found for the specified criteria.")
            return

        table = Table(
            "Timestamp",
            "Action",
            "User/Agent",
            "Risk",
            "Input Hash",
            "Output Hash",
            "Correlation ID",
        )
        for r in results:
            correlation_id_str = (
                r.metadata_.get("correlation_id") if r.metadata_ else "N/A"
            )
            table.add_row(
                str(r.ts),
                r.action,
                r.user_or_agent,
                r.risk_level,
                r.input_hash[:12] if r.input_hash else "N/A",
                r.output_hash[:12] if r.output_hash else "N/A",
                correlation_id_str,
            )
        console.print(table)

    except Exception as e:
        console.print(f"[red]Error querying audit log: {e}[/red]")
