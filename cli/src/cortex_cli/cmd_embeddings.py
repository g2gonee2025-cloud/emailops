"""
Embeddings subcommands for Cortex CLI.

Provides:
- `cortex embeddings stats` - Show embedding statistics
- `cortex embeddings backfill` - Backfill missing embeddings
"""

import argparse
import logging
import sys
import traceback
from typing import Any

from cortex_cli.style import colorize as _colorize

DEFAULT_BATCH_SIZE = 64
HIGH_COVERAGE_THRESHOLD = 95

logger = logging.getLogger(__name__)


def cmd_embeddings_stats(args: argparse.Namespace) -> None:
    """Show embedding statistics."""
    try:
        import json as json_module

        from cortex.config.loader import get_config
        from cortex.db.session import engine
        from sqlalchemy import text
        from sqlalchemy.orm import Session

        config = get_config()
        embed_cfg = getattr(config, "embedding", None)
        if not embed_cfg:
            print(
                f"{_colorize('ERROR:', 'red')} No embedding configuration found.",
                file=sys.stderr,
            )
            sys.exit(1)

        with Session(engine) as session:
            # Performance: Combined count query
            row = session.execute(
                text("SELECT COUNT(*), COUNT(embedding) FROM chunks")
            ).one()
            total = row[0] or 0
            with_emb = row[1] or 0

            missing = total - with_emb
            pct = (with_emb / total * 100) if total > 0 else 0

        stats = {
            "model": getattr(embed_cfg, "model_name", "N/A"),
            "dimensions": getattr(embed_cfg, "output_dimensionality", "N/A"),
            "total_chunks": total,
            "with_embedding": with_emb,
            "missing": missing,
            "coverage_pct": round(pct, 1),
        }

        if getattr(args, "json", False):
            print(json_module.dumps(stats, indent=2))
        else:
            print(f"\n{_colorize('EMBEDDING STATISTICS', 'bold')}\n")
            model_name = getattr(embed_cfg, "model_name", "N/A")
            dimensions = getattr(embed_cfg, "output_dimensionality", "N/A")
            print(f"  Model: {_colorize(model_name, 'cyan')}")
            print(f"  Dimensions: {_colorize(str(dimensions), 'cyan')}\n")
            print(f"  Total Chunks:    {_colorize(str(total), 'bold')}")
            print(f"  With Embedding:  {_colorize(str(with_emb), 'green')}")
            print(
                f"  Missing:         {_colorize(str(missing), 'yellow' if missing > 0 else 'green')}"
            )
            print(
                f"  Coverage:        {_colorize(f'{pct:.1f}%', 'green' if pct > HIGH_COVERAGE_THRESHOLD else 'yellow')}"
            )

            if missing > 0:
                print(
                    f"\n  {_colorize('TIP:', 'yellow')} Run `cortex embeddings backfill` to generate missing embeddings."
                )

    except Exception as e:
        logger.exception("Embeddings stats failed")
        print(f"{_colorize('ERROR:', 'red')} {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


def cmd_embeddings_backfill(args: argparse.Namespace) -> None:
    """Backfill missing embeddings."""
    try:
        from cortex.config.loader import get_config
        from cortex.ingestion.backfill import backfill_embeddings

        config = get_config()
        embed_cfg = getattr(config, "embedding", None)
        if not embed_cfg:
            print(
                f"{_colorize('ERROR:', 'red')} No embedding configuration found.",
                file=sys.stderr,
            )
            sys.exit(1)

        print(f"\n{_colorize('EMBEDDING BACKFILL', 'bold')}\n")
        print(f"  Model: {_colorize(embed_cfg.model_name, 'cyan')}")
        print(f"  Batch Size: {_colorize(str(args.batch_size), 'dim')}")

        if args.batch_size <= 0:
            print(
                f"{_colorize('ERROR:', 'red')} Batch size must be positive.",
                file=sys.stderr,
            )
            sys.exit(1)
        if args.limit is not None and args.limit < 0:
            print(
                f"{_colorize('ERROR:', 'red')} Limit must be zero or positive.",
                file=sys.stderr,
            )
            sys.exit(1)

        if args.dry_run:
            print("\n  DRY RUN: Would backfill embeddings. No changes made.")
            return

        print(f"\n  {_colorize('⏳', 'yellow')} Starting backfill...")

        result = backfill_embeddings(
            batch_size=args.batch_size,
            limit=args.limit,
        )

        print(f"\n  {_colorize('✓', 'green')} Backfill complete!")
        result_data = result if isinstance(result, dict) else {}
        if not isinstance(result, dict):
            logger.error("Unexpected backfill result type: %s", type(result))
        print(f"    Processed: {result_data.get('processed', 0)}")
        print(f"    Errors: {result_data.get('errors', 0)}")

    except ImportError as e:
        print(
            f"{_colorize('ERROR:', 'red')} Backfill module not available: {e}",
            file=sys.stderr,
        )
        print(
            "  Check that `cortex.ingestion.backfill` exists.",
            file=sys.stderr,
        )
        sys.exit(1)
    except Exception as e:
        logger.exception("Embeddings backfill failed")
        print(f"{_colorize('ERROR:', 'red')} {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
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
    stats_parser.add_argument("--json", action="store_true", help="Output as JSON")
    stats_parser.set_defaults(func=cmd_embeddings_stats)

    # embeddings backfill
    backfill_parser = emb_subparsers.add_parser(
        "backfill",
        help="Backfill missing embeddings",
        description="Generate embeddings for chunks that are missing them.",
    )
    backfill_parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Batch size (default: {DEFAULT_BATCH_SIZE})",
    )
    backfill_parser.add_argument(
        "--limit", "-l", type=int, default=None, help="Limit number to process"
    )
    backfill_parser.add_argument(
        "--dry-run", "-n", action="store_true", help="Show what would be done"
    )
    backfill_parser.set_defaults(func=cmd_embeddings_backfill)

    # Default: show stats when no subcommand given
    def _default_embeddings_handler(args: argparse.Namespace) -> None:
        if not getattr(args, "embeddings_command", None):
            args.json = getattr(args, "json", False)
            cmd_embeddings_stats(args)

    emb_parser.set_defaults(func=_default_embeddings_handler)
