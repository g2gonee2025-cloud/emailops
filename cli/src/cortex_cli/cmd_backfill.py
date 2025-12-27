
"""
Backfill subcommands for Cortex CLI.
"""
from __future__ import annotations
import argparse
import sys
from cortex_cli.style import colorize as _colorize
def backfill_summaries(args: argparse.Namespace) -> None:
    """Backfill summaries for conversations."""
    try:
        from scripts.backfill_summaries_simple import main as backfill_main
        # In a real scenario, you might refactor backfill_summaries_simple to be
        # a library, but for this example, we'll call its main function and
        # patch sys.argv.
        original_argv = sys.argv
        sys.argv = ["backfill_summaries_simple.py"]
        if args.tenant_id:
            sys.argv.extend(["--tenant-id", args.tenant_id])
        if args.limit:
            sys.argv.extend(["--limit", str(args.limit)])
        if args.workers:
            sys.argv.extend(["--workers", str(args.workers)])
        backfill_main()
        sys.argv = original_argv
    except ImportError as e:
        print(f"{_colorize('ERROR:', 'red')} Could not import backfill script: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"{_colorize('ERROR:', 'red')} {e}")
        sys.exit(1)
def setup_backfill_parser(subparsers: "argparse._SubParsersAction[argparse.ArgumentParser]") -> None:
    """Add backfill subcommands to the CLI parser."""
    backfill_parser = subparsers.add_parser(
        "backfill",
        help="Backfill data commands",
        description="Backfill data for conversations.",
    )
    backfill_subparsers = backfill_parser.add_subparsers(dest="backfill_command", title="Backfill Commands")
    # backfill summaries
    summaries_parser = backfill_subparsers.add_parser(
        "summaries",
        help="Backfill summaries for conversations",
        description="Generate summaries for conversations that are missing them.",
    )
    summaries_parser.add_argument("--tenant-id", type=str, help="Tenant ID to process")
    summaries_parser.add_argument("--limit", type=int, help="Limit the number of conversations to process")
    summaries_parser.add_argument("--workers", type=int, help="Number of workers to use")
    summaries_parser.set_defaults(func=backfill_summaries)
    def _default_backfill_handler(args: argparse.Namespace) -> None:
        if not args.backfill_command:
            backfill_parser.print_help()
    backfill_parser.set_defaults(func=_default_backfill_handler)
