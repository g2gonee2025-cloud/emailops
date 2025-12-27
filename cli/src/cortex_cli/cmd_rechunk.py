"""
CLI command for re-chunking oversized chunks.
"""

import argparse
from typing import Any

from cortex_cli.operations.rechunk import run_rechunk
from rich import box
from rich.console import Console
from rich.live import Live
from rich.progress import Progress, TextColumn
from rich.table import Table

console = Console()


def cmd_db_rechunk(args: argparse.Namespace) -> None:
    """Run the rechunking operation."""
    progress_table = Table.grid(expand=True)
    progress_table.add_row(
        "[bold cyan]Re-chunking Progress[/bold cyan]",
    )
    progress = Progress(TextColumn("{task.description}"), transient=True)
    progress_table.add_row(progress)

    results = {}
    task_id = progress.add_task("Starting re-chunking...", total=None)

    def _update_progress(res):
        total = res.get("total", 0)
        processed = res.get("processed", 0)
        new = res.get("new", 0)
        progress.update(
            task_id,
            description=f"Processed {processed}/{total} oversized chunks. Created {new} new chunks.",
            total=total,
            completed=processed,
        )

    with Live(progress_table, console=console, refresh_per_second=10):
        results = run_rechunk(
            tenant_id=args.tenant_id,
            chunk_size_limit=args.chunk_size_limit,
            dry_run=args.dry_run,
            max_tokens=args.max_tokens,
            progress_callback=_update_progress,
        )

    console.print("\n[bold green]Re-chunking Complete![/bold green]")
    summary_table = Table(title="Re-chunking Summary", box=box.ROUNDED)
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", style="bold", justify="right")
    for key, value in results.items():
        summary_table.add_row(key.replace("_", " ").title(), str(value))
    console.print(summary_table)


def setup_rechunk_parser(subparsers: Any) -> None:
    """Add rechunk subcommand to the CLI parser."""
    rechunk_parser = subparsers.add_parser(
        "rechunk",
        help="Re-chunk oversized chunks",
        description="Finds and re-chunks oversized chunks into smaller, valid chunks.",
    )
    rechunk_parser.add_argument("--tenant-id", type=str, help="Filter by tenant ID")
    rechunk_parser.add_argument(
        "--chunk-size-limit",
        type=int,
        default=8000,
        help="The character length limit for chunks (default: 8000)",
    )
    rechunk_parser.add_argument(
        "--dry-run", action="store_true", help="Don't write to DB"
    )
    rechunk_parser.add_argument(
        "--max-tokens",
        type=int,
        default=1600,
        help="The token limit for the chunker (default: 1600)",
    )
    rechunk_parser.set_defaults(func=cmd_db_rechunk)
