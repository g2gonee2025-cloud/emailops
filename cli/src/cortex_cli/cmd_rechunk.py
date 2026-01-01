"""
CLI command for re-chunking oversized chunks.
"""

import argparse
from collections.abc import Mapping
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

    task_id = progress.add_task("Starting re-chunking...", total=None)

    def _coerce_int(value: Any) -> int | None:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _update_progress(res: Mapping[str, Any] | None) -> None:
        if not isinstance(res, Mapping):
            return
        total = _coerce_int(res.get("total"))
        processed = _coerce_int(res.get("processed")) or 0
        new = _coerce_int(res.get("new")) or 0
        completed = processed if total is None else min(processed, total)
        if total is None:
            description = (
                f"Processed {processed} oversized chunks. Created {new} new chunks."
            )
        else:
            description = f"Processed {processed}/{total} oversized chunks. Created {new} new chunks."
        progress.update(
            task_id,
            description=description,
            total=total,
            completed=completed,
        )

    try:
        with Live(progress_table, console=console, refresh_per_second=10):
            results = run_rechunk(
                tenant_id=args.tenant_id,
                chunk_size_limit=args.chunk_size_limit,
                dry_run=args.dry_run,
                max_tokens=args.max_tokens,
                progress_callback=_update_progress,
            )
    except Exception as exc:
        console.print(f"[red]Re-chunking failed: {exc}[/red]")
        raise SystemExit(1)

    if not isinstance(results, Mapping):
        console.print("[red]Unexpected rechunk results; expected a summary map.[/red]")
        raise SystemExit(1)

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
