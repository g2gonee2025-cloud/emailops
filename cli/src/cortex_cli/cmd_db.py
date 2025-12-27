"""
Database subcommands for Cortex CLI.
Provides:
- `cortex db stats` - Show database statistics
- `cortex db migrate` - Run Alembic migrations
- `cortex db cleanup` - Run database cleanup tasks
- `cortex db backfill-graph` - Backfill graph data
"""

import argparse
import subprocess
import sys
from pathlib import Path
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
        from cortex_cli.config import get_config
        from cortex_cli.utils import _colorize
        from sqlalchemy import text
        from sqlalchemy.orm import Session


        config = get_config()

        if not args.json:
            print(f"\n{_colorize('DATABASE STATISTICS', 'bold')}\n")
            print(
                f"  Database: {_colorize(config.database.url.split('@')[-1] if '@' in config.database.url else 'local', 'cyan')}\n"
            )

        with Session(engine) as session:
            queries = {
                "Conversations": "SELECT COUNT(*) FROM conversations",
                "Attachments": "SELECT COUNT(*) FROM attachments",
                "Chunks (Total)": "SELECT COUNT(*) FROM chunks",
                "Chunks (With Embedding)": "SELECT COUNT(*) FROM chunks WHERE embedding IS NOT NULL",
                "Chunks (Missing Embedding)": "SELECT COUNT(*) FROM chunks WHERE embedding IS NULL",
                "Entity Nodes": "SELECT COUNT(*) FROM entity_nodes",
                "Entity Edges": "SELECT COUNT(*) FROM entity_edges",
            }

            stats = {}
            for label, query in queries.items():
                try:
                    result = session.execute(text(query)).scalar()
                    stats[label] = result or 0
                except Exception:
                    stats[label] = "N/A"

        if args.json:
            import json

            print(json.dumps(stats, indent=2))
        elif RICH_AVAILABLE and console:
            table = Table(title="Database Stats", box=box.ROUNDED)
            table.add_column("Metric", style="cyan")
            table.add_column("Count", style="bold")
            for k, v in stats.items():
                table.add_row(k, str(v))
            console.print(table)
        else:
            for k, v in stats.items():
                print(f"  {k}: {_colorize(str(v), 'green')}")

        total = stats.get("Chunks (Total)", 0)
        with_emb = stats.get("Chunks (With Embedding)", 0)
        if isinstance(total, int) and isinstance(with_emb, int) and total > 0:
            pct = (with_emb / total) * 100
            print(
                f"\n  Embedding Coverage: {_colorize(f'{pct:.1f}%', 'green' if pct > 90 else 'yellow')}"
            )

>>>>>>> origin/main
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
    backend_dir = Path(__file__).resolve().parents[3] / "backend"
    migrations_dir = backend_dir / "migrations"

    if not migrations_dir.exists():
        print(f"{_colorize('ERROR:', 'red')} Migrations directory not found: {migrations_dir}")
        sys.exit(1)

    cmd = ["alembic", "-c", "migrations/alembic.ini", "upgrade", "head"]
    if args.dry_run:
        cmd.extend(["--sql"])

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


def cmd_db_cleanup(args: argparse.Namespace) -> None:
    """Run the database cleanup script."""
    script_path = Path(__file__).resolve().parents[3] / "scripts" / "run_db_cleanup.py"
    if not script_path.exists():
        print(f"{_colorize('ERROR:', 'red')} Cleanup script not found at {script_path}")
        sys.exit(1)

    cmd = [sys.executable, str(script_path)]
    if args.tenant_id:
        cmd.extend(["--tenant-id", args.tenant_id])
    elif args.all_tenants:
        cmd.append("--all-tenants")

    if args.dry_run:
        cmd.append("--dry-run")

    print(f"{_colorize('▶ RUNNING DB CLEANUP', 'bold')}")
    print(f"  Command: {' '.join(cmd)}")
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
        )
        for line in iter(process.stdout.readline, ""):
            print(f"  {line.strip()}")
        process.wait()
        if process.returncode == 0:
            print(f"\n{_colorize('✓', 'green')} Cleanup script finished successfully.")
        else:
            print(f"\n{_colorize('✗', 'red')} Cleanup script failed (exit code {process.returncode}).")
            sys.exit(process.returncode)
    except FileNotFoundError:
        print(f"{_colorize('ERROR:', 'red')} Could not execute Python at: {sys.executable}")
        sys.exit(1)
    except Exception as e:
        print(f"{_colorize('ERROR:', 'red')} An unexpected error occurred: {e}")
        sys.exit(1)


def setup_db_parser(subparsers: Any) -> None:
    """Add db subcommands to the CLI parser."""
    db_parser = subparsers.add_parser(
        "db",
        help="Database management commands",
        description="Manage the Cortex database.",
    )

    db_subparsers = db_parser.add_subparsers(dest="db_command", title="DB Commands")

    # db stats
    stats_parser = db_subparsers.add_parser(
        "stats", help="Show database statistics"
    )
    stats_parser.add_argument("--json", action="store_true", help="Output as JSON")
    stats_parser.set_defaults(func=cmd_db_stats)

    # db migrate
    migrate_parser = db_subparsers.add_parser(
        "migrate", help="Run Alembic migrations"
    )
    migrate_parser.add_argument(
        "--dry-run", action="store_true", help="Show SQL without executing"
    )
    migrate_parser.set_defaults(func=cmd_db_migrate)

    # db backfill-graph
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

    # db cleanup
    cleanup_parser = db_subparsers.add_parser(
        "cleanup", help="Run database cleanup tasks"
    )
    group = cleanup_parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--tenant-id", type=str, help="Specific tenant ID to clean up.")
    group.add_argument(
        "--all-tenants", action="store_true", help="Run cleanup for all tenants."
    )
    cleanup_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate cleanup without making changes.",
    )
    cleanup_parser.set_defaults(func=cmd_db_cleanup)

    def _default_db_handler(args: argparse.Namespace) -> None:
        if not args.db_command:
            args.json = getattr(args, "json", False)
            cmd_db_stats(args)

    db_parser.set_defaults(func=_default_db_handler)
