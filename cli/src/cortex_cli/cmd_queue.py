"""
Queue CLI commands.
"""

from cortex.queue import get_queue
from rich.console import Console
from rich.table import Table

console = Console()


def cmd_queue_stats() -> None:
    """Display queue statistics."""
    q = get_queue()
    stats = q.get_queue_stats()

    table = Table(title="Queue Statistics")
    table.add_column("Metric", style="dim")
    table.add_column("Value", style="bold")

    for key, value in stats.items():
        table.add_row(key.replace("_", " ").title(), str(value))

    console.print(table)


def setup_queue_parser(subparsers):
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
    stats_parser.set_defaults(func=lambda _: cmd_queue_stats())
