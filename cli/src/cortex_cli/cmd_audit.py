"""
Audit log command.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from typing import Any, Protocol

from cortex.audit import get_audit_log_cli


class _Subparsers(Protocol):
    def add_parser(self, *args: Any, **kwargs: Any) -> argparse.ArgumentParser: ...


def _parse_since(value: str | None) -> datetime | None:
    if not value:
        return None
    raw = value.strip()
    if raw.endswith("Z"):
        raw = f"{raw[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError as exc:
        raise ValueError(
            "Invalid --since timestamp; expected ISO-8601 format."
        ) from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def setup_audit_parser(
    subparsers: _Subparsers,
) -> None:
    """Setup audit subcommand parser."""
    parser = subparsers.add_parser("audit", help="View audit logs")
    parser.set_defaults(func=audit_log)
    parser.add_argument("tenant_id", help="Tenant ID for the audit log")
    parser.add_argument(
        "--limit", type=int, default=100, help="Number of records to return"
    )
    parser.add_argument("--since", type=str, help="Start time (ISO format)")
    parser.add_argument("--user", type=str, help="Filter by user or agent")
    parser.add_argument("--action", type=str, help="Filter by action")
    parser.add_argument("--correlation-id", type=str, help="Filter by correlation ID")


def audit_log(args: argparse.Namespace) -> None:
    """Get audit log."""
    try:
        since = _parse_since(getattr(args, "since", None))
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)

    limit = getattr(args, "limit", 100)
    if not isinstance(limit, int) or limit <= 0:
        print("Error: --limit must be a positive integer.", file=sys.stderr)
        raise SystemExit(1)
    if limit > 1000:
        print("Error: --limit must be 1000 or less.", file=sys.stderr)
        raise SystemExit(1)

    try:
        get_audit_log_cli(
            tenant_id=args.tenant_id,
            limit=limit,
            since=since,
            user_or_agent=args.user,
            action=args.action,
            correlation_id=args.correlation_id,
        )
    except Exception as exc:
        print(f"Error querying audit log: {exc}", file=sys.stderr)
        raise SystemExit(1)
