"""
S3/Spaces subcommands for Cortex CLI.

Provides:
- `cortex s3 list` - List S3 prefixes/folders
- `cortex s3 ingest` - Ingest from S3
"""
import argparse
import sys
from typing import Any

try:
    from rich import box
    from rich.console import Console
    from rich.table import Table

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

console = Console() if RICH_AVAILABLE else None


def _colorize(text: str, color: str) -> str:
    colors = {
        "reset": "\033[0m",
        "bold": "\033[1m",
        "dim": "\033[2m",
        "green": "\033[32m",
        "yellow": "\033[33m",
        "cyan": "\033[36m",
        "red": "\033[31m",
    }
    if not sys.stdout.isatty():
        return text
    return f"{colors.get(color, '')}{text}{colors['reset']}"


def cmd_s3_list(args: argparse.Namespace) -> None:
    """List S3/Spaces prefixes (conversation folders)."""
    try:
        from cortex.config.loader import get_config
        from cortex.ingestion.s3_source import S3SourceHandler

        config = get_config()
        handler = S3SourceHandler()

        print(f"\n{_colorize('S3/SPACES BROWSER', 'bold')}\n")
        print(f"  Endpoint: {_colorize(config.storage.endpoint_url, 'cyan')}")
        print(f"  Bucket:   {_colorize(config.storage.bucket_raw, 'cyan')}")
        if args.prefix:
            print(f"  Prefix:   {_colorize(args.prefix, 'dim')}")
        print()

        folders = list(handler.list_conversation_folders(prefix=args.prefix or ""))

        if not folders:
            print(f"  {_colorize('○', 'yellow')} No conversation folders found.")
            return

        if args.json:
            import json

            # Serialize folder objects to dicts
            folder_data = [{"name": f.name, "prefix": f.prefix} for f in folders]
            print(json.dumps({"folders": folder_data}, indent=2))
        elif RICH_AVAILABLE and console:
            table = Table(title=f"Folders ({len(folders)})", box=box.SIMPLE)
            table.add_column("#", style="dim")
            table.add_column("Prefix", style="cyan")
            for i, folder in enumerate(folders[: args.limit or 50], 1):
                table.add_row(str(i), folder.name)
            if len(folders) > (args.limit or 50):
                table.add_row("...", f"({len(folders) - (args.limit or 50)} more)")
            console.print(table)
        else:
            for i, folder in enumerate(folders[: args.limit or 50], 1):
                print(f"  {i}. {folder.name}")
            if len(folders) > (args.limit or 50):
                print(f"  ... and {len(folders) - (args.limit or 50)} more")

    except Exception as e:
        print(f"{_colorize('ERROR:', 'red')} {e}")
        sys.exit(1)


def cmd_s3_ingest(args: argparse.Namespace) -> None:
    """Ingest conversations from S3/Spaces."""
    import uuid

    try:
        from cortex.config.loader import get_config
        from cortex.ingestion.mailroom import process_job
        from cortex.ingestion.models import IngestJobRequest
        from cortex.ingestion.s3_source import S3SourceHandler

        config = get_config()
        handler = S3SourceHandler()

        print(f"\n{_colorize('S3/SPACES INGESTION', 'bold')}\n")
        print(f"  Prefix:  {_colorize(args.prefix, 'cyan')}")
        print(f"  Tenant:  {_colorize(args.tenant, 'dim')}")

        if args.dry_run:
            # List what would be ingested
            folders = list(handler.list_conversation_folders(prefix=args.prefix))
            print(
                f"\n  {_colorize('DRY RUN:', 'yellow')} Would ingest {len(folders)} folder(s):"
            )
            for f in folders[:10]:
                print(f"    • {f.name}")
            if len(folders) > 10:
                print(f"    ... and {len(folders) - 10} more")
            return

        # Ingest
        folders = list(handler.list_conversation_folders(prefix=args.prefix))
        if not folders:
            print(
                f"\n  {_colorize('⚠', 'yellow')} No folders found with prefix: {args.prefix}"
            )
            return

        print(f"\n  {_colorize('⏳', 'yellow')} Ingesting {len(folders)} folder(s)...\n")

        success = 0
        failed = 0

        for i, folder in enumerate(folders, 1):
            print(f"  [{i}/{len(folders)}] {folder.name}...", end=" ", flush=True)

            job = IngestJobRequest(
                job_id=uuid.uuid4(),
                tenant_id=args.tenant,
                source_type="s3",
                source_uri=f"s3://{config.storage.bucket_raw}/{folder.prefix}",
            )

            try:
                summary = process_job(job)
                if summary.aborted_reason:
                    print(f"{_colorize('FAILED', 'red')}")
                    failed += 1
                else:
                    print(
                        f"{_colorize('OK', 'green')} ({summary.messages_ingested} msg)"
                    )
                    success += 1
            except Exception as e:
                print(f"{_colorize('ERROR', 'red')}: {e}")
                failed += 1

        print(f"\n{_colorize('═' * 40, 'cyan')}")
        print(f"  Succeeded: {_colorize(str(success), 'green')}")
        print(f"  Failed:    {_colorize(str(failed), 'red' if failed > 0 else 'dim')}")

    except Exception as e:
        print(f"{_colorize('ERROR:', 'red')} {e}")
        sys.exit(1)


def setup_s3_parser(subparsers: Any) -> None:
    """Add s3 subcommands to the CLI parser."""
    s3_parser = subparsers.add_parser(
        "s3",
        help="S3/Spaces storage commands",
        description="Browse and ingest from S3/DigitalOcean Spaces.",
    )

    s3_subparsers = s3_parser.add_subparsers(dest="s3_command", title="S3 Commands")

    # s3 list
    list_parser = s3_subparsers.add_parser(
        "list",
        help="List S3 prefixes/folders",
        description="Browse conversation folders in S3/Spaces.",
    )
    list_parser.add_argument("--prefix", "-p", default="", help="S3 prefix to filter")
    list_parser.add_argument(
        "--limit", "-l", type=int, default=50, help="Max folders to show"
    )
    list_parser.add_argument("--json", action="store_true", help="Output as JSON")
    list_parser.set_defaults(func=cmd_s3_list)

    # s3 ingest
    ingest_parser = s3_subparsers.add_parser(
        "ingest",
        help="Ingest from S3/Spaces",
        description="Ingest conversation folders from S3/Spaces.",
    )
    ingest_parser.add_argument("prefix", metavar="PREFIX", help="S3 prefix to ingest")
    ingest_parser.add_argument("--tenant", "-t", default="default", help="Tenant ID")
    ingest_parser.add_argument(
        "--dry-run", "-n", action="store_true", help="Show what would be done"
    )
    ingest_parser.set_defaults(func=cmd_s3_ingest)
