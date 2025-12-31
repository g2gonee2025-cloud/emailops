"""
Queue CLI commands.
"""

import argparse
from typing import Any, Protocol

from cortex.queue import get_queue
from rich.console import Console
from rich.table import Table

console = Console()


class _Subparsers(Protocol):
    def add_parser(self, *args: Any, **kwargs: Any) -> argparse.ArgumentParser: ...


def cmd_queue_stats(_args: argparse.Namespace) -> None:
    """Display queue statistics."""
    try:
        q = get_queue()
    except Exception as exc:
        console.print(f"[red]Failed to load queue: {exc}[/red]")
        raise SystemExit(1)

    if q is None:
        console.print("[red]Queue is not configured.[/red]")
        raise SystemExit(1)

    stats = q.get_queue_stats()
    if not isinstance(stats, dict):
        console.print("[red]Queue stats unavailable.[/red]")
        raise SystemExit(1)

    table = Table(title="Queue Statistics")
    table.add_column("Metric", style="dim")
    table.add_column("Value", style="bold")

    for key, value in stats.items():
        key_label = str(key).replace("_", " ").title()
        table.add_row(key_label, str(value))

    console.print(table)


def setup_queue_parser(subparsers: _Subparsers) -> None:
    """Setup queue command parser."""
    queue_parser = subparsers.add_parser(
        "queue",
        help="Job queue management",
        description="Inspect and manage the job queue.",
    )
    queue_subparsers = queue_parser.add_subparsers(
        dest="queue_command",
        title="Queue Commands",
    )

    # stats
    stats_parser = queue_subparsers.add_parser(
        "stats",
        help="Show queue statistics",
    )
    stats_parser.set_defaults(func=cmd_queue_stats)

    def _default_queue_handler(_args: argparse.Namespace) -> None:
        queue_parser.print_help()
        raise SystemExit(1)

    queue_parser.set_defaults(func=_default_queue_handler)
