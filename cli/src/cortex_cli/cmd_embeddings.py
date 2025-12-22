"""
Embeddings subcommands for Cortex CLI.

Provides:
- `cortex embeddings stats` - Show embedding statistics
- `cortex embeddings backfill` - Backfill missing embeddings
"""

import argparse
import sys
from typing import Any

try:
    from rich.console import Console

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


def cmd_embeddings_stats(_args: argparse.Namespace) -> None:
    """Show embedding statistics."""
    try:
        from cortex.config.loader import get_config
        from cortex.db.session import engine
        from sqlalchemy import text
        from sqlalchemy.orm import Session

        config = get_config()
        # engine is already imported

        print(f"\n{_colorize('EMBEDDING STATISTICS', 'bold')}\n")
        print(f"  Model: {_colorize(config.embedding.model_name, 'cyan')}")
        print(
            f"  Dimensions: {_colorize(str(config.embedding.output_dimensionality), 'cyan')}\n"
        )

        with Session(engine) as session:
            total = session.execute(text("SELECT COUNT(*) FROM chunks")).scalar() or 0
            with_emb = (
                session.execute(
                    text("SELECT COUNT(*) FROM chunks WHERE embedding IS NOT NULL")
                ).scalar()
                or 0
            )
            missing = total - with_emb

            pct = (with_emb / total * 100) if total > 0 else 0

            print(f"  Total Chunks:    {_colorize(str(total), 'bold')}")
            print(f"  With Embedding:  {_colorize(str(with_emb), 'green')}")
            print(
                f"  Missing:         {_colorize(str(missing), 'yellow' if missing > 0 else 'green')}"
            )
            print(
                f"  Coverage:        {_colorize(f'{pct:.1f}%', 'green' if pct > 95 else 'yellow')}"
            )

            if missing > 0:
                print(
                    f"\n  {_colorize('TIP:', 'yellow')} Run `cortex embeddings backfill` to generate missing embeddings."
                )

    except Exception as e:
        print(f"{_colorize('ERROR:', 'red')} {e}")
        sys.exit(1)


def cmd_embeddings_backfill(args: argparse.Namespace) -> None:
    """Backfill missing embeddings."""
    try:
        from cortex.config.loader import get_config
        from cortex.ingestion.backfill import backfill_embeddings

        config = get_config()

        print(f"\n{_colorize('EMBEDDING BACKFILL', 'bold')}\n")
        print(f"  Model: {_colorize(config.embedding.model_name, 'cyan')}")
        print(f"  Batch Size: {_colorize(str(args.batch_size), 'dim')}")

        if args.dry_run:
            print("\n  DRY RUN: Would backfill embeddings. No changes made.")
            return

        print(f"\n  {_colorize('⏳', 'yellow')} Starting backfill...")

        result = backfill_embeddings(
            batch_size=args.batch_size,
            limit=args.limit,
        )

        print(f"\n  {_colorize('✓', 'green')} Backfill complete!")
        print(f"    Processed: {result.get('processed', 0)}")
        print(f"    Errors: {result.get('errors', 0)}")

    except ImportError as e:
        print(f"{_colorize('ERROR:', 'red')} Backfill module not available: {e}")
        print("  Check that `cortex.ingestion.backfill` exists.")
        sys.exit(1)
    except Exception as e:
        print(f"{_colorize('ERROR:', 'red')} {e}")
        sys.exit(1)


def setup_embeddings_parser(subparsers: Any) -> None:
    """Add embeddings subcommands to the CLI parser."""
    emb_parser = subparsers.add_parser(
        "embeddings",
        help="Embedding management commands",
        description="Manage embeddings: view stats, backfill missing.",
    )

    emb_subparsers = emb_parser.add_subparsers(
        dest="embeddings_command", title="Embedding Commands"
    )

    # embeddings stats
    stats_parser = emb_subparsers.add_parser(
        "stats",
        help="Show embedding statistics",
        description="Display embedding model info and coverage stats.",
    )
    stats_parser.set_defaults(func=cmd_embeddings_stats)

    # embeddings backfill
    backfill_parser = emb_subparsers.add_parser(
        "backfill",
        help="Backfill missing embeddings",
        description="Generate embeddings for chunks that are missing them.",
    )
    backfill_parser.add_argument(
        "--batch-size", "-b", type=int, default=64, help="Batch size (default: 64)"
    )
    backfill_parser.add_argument(
        "--limit", "-l", type=int, default=None, help="Limit number to process"
    )
    backfill_parser.add_argument(
        "--dry-run", "-n", action="store_true", help="Show what would be done"
    )
    backfill_parser.set_defaults(func=cmd_embeddings_backfill)
