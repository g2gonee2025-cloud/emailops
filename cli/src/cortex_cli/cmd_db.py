"""
Database subcommands for Cortex CLI.

Provides:
- `cortex db stats` - Show database statistics
- `cortex db migrate` - Run Alembic migrations
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
    """Fallback colorize if rich not available."""
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


def cmd_db_stats(args: argparse.Namespace) -> None:
    """Show database statistics: counts for threads, messages, chunks, embeddings."""
    try:
        from cortex.config.loader import get_config
        from cortex.db.session import engine
        from sqlalchemy import text
        from sqlalchemy.orm import Session

        config = get_config()

        if not args.json:
            print(f"\n{_colorize('DATABASE STATISTICS', 'bold')}\n")
            print(
                f"  Database: {_colorize(config.database.url.split('@')[-1] if '@' in config.database.url else 'local', 'cyan')}\n"
            )

        # engine is already imported

        with Session(engine) as session:
            # Get counts - using actual table names from schema
            queries = {
                "Conversations": "SELECT COUNT(*) FROM conversations",
                "Attachments": "SELECT COUNT(*) FROM attachments",
                "Chunks (Total)": "SELECT COUNT(*) FROM chunks",
                "Chunks (With Embedding)": "SELECT COUNT(*) FROM chunks WHERE embedding IS NOT NULL",
                "Chunks (Missing Embedding)": "SELECT COUNT(*) FROM chunks WHERE embedding IS NULL",
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

        # Embedding coverage
        total = stats.get("Chunks (Total)", 0)
        with_emb = stats.get("Chunks (With Embedding)", 0)
        if isinstance(total, int) and isinstance(with_emb, int) and total > 0:
            pct = (with_emb / total) * 100
            print(
                f"\n  Embedding Coverage: {_colorize(f'{pct:.1f}%', 'green' if pct > 90 else 'yellow')}"
            )

    except ImportError as e:
        print(f"{_colorize('ERROR:', 'red')} Could not import database modules: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"{_colorize('ERROR:', 'red')} {e}")
        sys.exit(1)


def cmd_db_migrate(args: argparse.Namespace) -> None:
    """Run Alembic migrations."""
    import subprocess
    from pathlib import Path

    backend_dir = Path(__file__).resolve().parents[3] / "backend"
    migrations_dir = backend_dir / "migrations"

    if not migrations_dir.exists():
        print(
            f"{_colorize('ERROR:', 'red')} Migrations directory not found: {migrations_dir}"
        )
        sys.exit(1)

    cmd = ["alembic", "upgrade", "head"]
    if args.dry_run:
        cmd = ["alembic", "upgrade", "head", "--sql"]
        print(f"{_colorize('DRY RUN:', 'yellow')} Would run: {' '.join(cmd)}\n")

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(backend_dir), capture_output=not args.verbose)

    if result.returncode == 0:
        print(f"\n{_colorize('✓', 'green')} Migrations applied successfully.")
    else:
        print(f"\n{_colorize('✗', 'red')} Migration failed.")
        if result.stderr:
            print(result.stderr.decode())
        sys.exit(1)


def setup_db_parser(subparsers: Any) -> None:
    """Add db subcommands to the CLI parser."""
    db_parser = subparsers.add_parser(
        "db",
        help="Database management commands",
        description="Manage the Cortex database: view stats, run migrations.",
    )

    db_subparsers = db_parser.add_subparsers(dest="db_command", title="DB Commands")

    # db stats
    stats_parser = db_subparsers.add_parser(
        "stats",
        help="Show database statistics",
        description="Display counts for threads, messages, chunks, and embedding coverage.",
    )
    stats_parser.add_argument("--json", action="store_true", help="Output as JSON")
    stats_parser.set_defaults(func=cmd_db_stats)

    # db migrate
    migrate_parser = db_subparsers.add_parser(
        "migrate",
        help="Run Alembic migrations",
        description="Apply pending database migrations.",
    )
    migrate_parser.add_argument(
        "--dry-run", "-n", action="store_true", help="Show SQL without executing"
    )
    migrate_parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show migration output"
    )
    migrate_parser.set_defaults(func=cmd_db_migrate)
