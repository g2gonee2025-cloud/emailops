"""
Audit log command.
"""

from __future__ import annotations

import argparse
from datetime import datetime

from cortex.audit import get_audit_log_cli


def setup_audit_parser(parser: argparse.ArgumentParser) -> None:
    """Setup audit subcommand parser."""
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
    since = datetime.fromisoformat(args.since) if args.since else None
    get_audit_log_cli(
        tenant_id=args.tenant_id,
        limit=args.limit,
        since=since,
        user_or_agent=args.user,
        action=args.action,
        correlation_id=args.correlation_id,
    )
