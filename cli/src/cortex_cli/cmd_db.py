"""
Database subcommands for Cortex CLI.
"""

import argparse
import sys
from typing import Any

from cortex_cli.operations.backfill_graph import run_backfill_graph

try:
    from rich import box
    from rich.console import Console
    from rich.live import Live
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.table import Table

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

console = Console() if RICH_AVAILABLE else None


def cmd_db_stats(args: argparse.Namespace) -> None:
    """Show database statistics."""
    try:
        from cortex.db.session import engine
        from sqlalchemy import text
        from sqlalchemy.orm import Session
    except ImportError as e:
        console.print(f"[red]Error:[/] Could not import database modules: {e}")
        sys.exit(1)

    with Session(engine) as session:
        queries = {
            "Conversations": "SELECT COUNT(*) FROM conversations",
            "Attachments": "SELECT COUNT(*) FROM attachments",
            "Chunks": "SELECT COUNT(*) FROM chunks",
            "Entity Nodes": "SELECT COUNT(*) FROM entity_nodes",
            "Entity Edges": "SELECT COUNT(*) FROM entity_edges",
        }
        stats = {
            label: session.execute(text(query)).scalar_one_or_none() or 0
            for label, query in queries.items()
        }

    if args.json:
        import json

        print(json.dumps(stats, indent=2))
        return

    table = Table(title="Database Statistics", box=box.ROUNDED)
    table.add_column("Metric", style="cyan")
    table.add_column("Count", style="bold", justify="right")
    for k, v in stats.items():
        table.add_row(k, f"{v:,}")
    console.print(table)


def cmd_db_backfill_graph(args: argparse.Namespace) -> None:
    """Run graph backfill operation."""
    if not RICH_AVAILABLE:
        print("Rich library not available. Please install it.")
        sys.exit(1)

    progress_table = Table.grid(expand=True)
    progress_table.add_row(
        "[bold cyan]Total Progress[/bold cyan]",
        "[bold magenta]Batch Progress[/bold magenta]",
    )
    total_progress = Progress(
        TextColumn("{task.description}"), transient=True
    )
    batch_progress = Progress(SpinnerColumn(), TextColumn("{task.description}"))
    progress_table.add_row(total_progress, batch_progress)

    results = {}
    task_id = total_progress.add_task("Starting backfill...", total=None)

    def _update_progress(res):
        total = res.get("total", 0)
        processed = res.get("success", 0) + res.get("failed", 0)
        total_progress.update(
            task_id,
            description=f"Processed {processed}/{total} conversations",
            total=total,
            completed=processed,
        )
        batch_progress.update(
            batch_task_id,
            description=f"Nodes: {res.get('nodes_created', 0)} | Edges: {res.get('edges_created', 0)}",
        )

    with Live(progress_table, console=console, refresh_per_second=10):
        batch_task_id = batch_progress.add_task("Running...", total=None)
        results = run_backfill_graph(
            tenant_id=args.tenant_id,
            max_workers=args.workers,
            limit=args.limit,
            dry_run=args.dry_run,
            progress_callback=_update_progress,
        )

    console.print("\n[bold green]Backfill Complete![/bold green]")
    summary_table = Table(title="Backfill Summary", box=box.ROUNDED)
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", style="bold", justify="right")
    for key, value in results.items():
        summary_table.add_row(key.replace("_", " ").title(), str(value))
    console.print(summary_table)


def cmd_db_migrate(args: argparse.Namespace) -> None:
    """Run Alembic migrations."""
    import subprocess
    from pathlib import Path

    backend_dir = Path(__file__).resolve().parents[3] / "backend"
    if not backend_dir.exists():
        console.print(f"[red]Error:[/] Backend directory not found at {backend_dir}")
        sys.exit(1)

    cmd = ["alembic", "-c", "migrations/alembic.ini", "upgrade", "head"]
    if args.dry_run:
        cmd.append("--sql")

    console.print(f"Running command: [cyan]{' '.join(cmd)}[/cyan]")
    result = subprocess.run(
        cmd, cwd=str(backend_dir), capture_output=True, text=True
    )

    if result.returncode == 0:
        console.print("[green]✓ Migrations applied successfully.[/green]")
        if args.dry_run:
            console.print(result.stdout)
    else:
        console.print(f"[red]✗ Migration failed.[/red]")
        console.print(result.stderr)
        sys.exit(1)


def setup_db_parser(subparsers: Any) -> None:
    """Add db subcommands to the CLI parser."""
    db_parser = subparsers.add_parser(
        "db", help="Database management commands"
    )
    db_subparsers = db_parser.add_subparsers(
        dest="db_command", title="DB Commands", required=True
    )

    stats_parser = db_subparsers.add_parser("stats", help="Show database statistics")
    stats_parser.add_argument("--json", action="store_true", help="Output as JSON")
    stats_parser.set_defaults(func=cmd_db_stats)

    migrate_parser = db_subparsers.add_parser("migrate", help="Run Alembic migrations")
    migrate_parser.add_argument(
        "--dry-run", action="store_true", help="Show SQL without executing"
    )
    migrate_parser.set_defaults(func=cmd_db_migrate)

    backfill_graph_parser = db_subparsers.add_parser(
        "backfill-graph", help="Backfill graph data from conversation summaries"
    )
    backfill_graph_parser.add_argument(
        "--tenant-id", type=str, help="Filter by tenant ID"
    )
    backfill_graph_parser.add_argument(
        "--workers", type=int, default=10, help="Number of parallel workers"
    )
    backfill_graph_parser.add_argument(
        "--limit", type=int, help="Limit number of conversations"
    )
    backfill_graph_parser.add_argument(
        "--dry-run", action="store_true", help="Don't write to DB"
    )
    backfill_graph_parser.set_defaults(func=cmd_db_backfill_graph)
