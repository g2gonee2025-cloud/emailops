"""
Backfill subcommands for Cortex CLI.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from typing import Any

from cortex_cli.style import colorize as _colorize


def backfill_summaries(args: argparse.Namespace) -> None:
    """Backfill summaries for conversations."""
    try:
        from scripts.backfill_summaries_simple import main as backfill_main
    except ImportError as e:
        print(
            f"{_colorize('ERROR:', 'red')} Could not import backfill script: {e}",
            file=sys.stderr,
        )
        traceback.print_exc()
        sys.exit(1)

    argv: list[str] = []
    tenant_id = getattr(args, "tenant_id", None)
    limit = getattr(args, "limit", None)
    workers = getattr(args, "workers", None)
    if tenant_id:
        argv.extend(["--tenant-id", tenant_id])
    if limit is not None:
        argv.extend(["--limit", str(limit)])
    if workers is not None:
        argv.extend(["--workers", str(workers)])

    try:
        backfill_main(argv)
    except SystemExit:
        raise
    except Exception:
        print(f"{_colorize('ERROR:', 'red')} Backfill failed.", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)


def setup_backfill_parser(
    subparsers: Any,
) -> None:
    """Add backfill subcommands to the CLI parser."""
    backfill_parser = subparsers.add_parser(
        "backfill",
        help="Backfill data commands",
        description="Backfill data for conversations.",
    )
    backfill_subparsers = backfill_parser.add_subparsers(
        dest="backfill_command", title="Backfill Commands"
    )
    # backfill summaries
    summaries_parser = backfill_subparsers.add_parser(
        "summaries",
        help="Backfill summaries for conversations",
        description="Generate summaries for conversations that are missing them.",
    )
    summaries_parser.add_argument("--tenant-id", type=str, help="Tenant ID to process")
    summaries_parser.add_argument(
        "--limit", type=int, help="Limit the number of conversations to process"
    )
    summaries_parser.add_argument(
        "--workers", type=int, help="Number of workers to use"
    )
    summaries_parser.set_defaults(func=backfill_summaries)

    def _default_backfill_handler(args: argparse.Namespace) -> None:
        if not args.backfill_command:
            backfill_parser.print_help()

    backfill_parser.set_defaults(func=_default_backfill_handler)
