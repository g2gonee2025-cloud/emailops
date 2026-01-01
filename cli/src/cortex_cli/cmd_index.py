"""
Index subcommands for Cortex CLI.
Provides:
- `cortex index stats` - Show index statistics
"""

import argparse
import sys
from typing import Any

from cortex_cli.style import colorize as _colorize

from .indexing.metadata import get_index_info


def cmd_index_stats(args: argparse.Namespace) -> None:
    """Show index statistics."""
    try:
        import json as json_module

        index_dir = args.index_dir
        info = get_index_info(index_dir, show_faiss_count=True)
        if not isinstance(info, dict):
            raise ValueError("Index metadata is unavailable or malformed.")

        def _format_value(value: object) -> str:
            return "N/A" if value is None else str(value)

        if getattr(args, "json", False):
            print(json_module.dumps(info, indent=2))
        else:
            print(f"\n{_colorize('INDEX STATISTICS', 'bold')}\n")
            print(f"  Index Directory: {_colorize(index_dir, 'cyan')}")

            metadata = info.get("metadata")
            if metadata == "unknown" or not isinstance(metadata, dict):
                print(f"  Metadata: {_colorize('Not found', 'red')}")
            else:
                print(
                    f"  Provider: {_colorize(_format_value(metadata.get('provider')), 'cyan')}"
                )
                print(
                    f"  Model: {_colorize(_format_value(metadata.get('model')), 'cyan')}"
                )
                print(
                    f"  Dimensions: {_colorize(_format_value(metadata.get('dimensions')), 'cyan')}"
                )
                print(
                    f"  Actual Dimensions: {_colorize(_format_value(metadata.get('actual_dimensions')), 'cyan')}"
                )
                print(
                    f"  Documents: {_colorize(_format_value(metadata.get('num_documents')), 'cyan')}"
                )

            dimensions = info.get("dimensions")
            if not isinstance(dimensions, dict):
                dimensions = {}
            print(f"\n{_colorize('Detected Properties', 'bold')}\n")
            print(f"  Index Type: {_colorize(info.get('index_type', 'N/A'), 'cyan')}")
            print(
                f"  Embeddings Dimensions: {_colorize(_format_value(dimensions.get('embeddings')), 'cyan')}"
            )
            print(
                f"  FAISS Dimensions: {_colorize(_format_value(dimensions.get('faiss')), 'cyan')}"
            )
            if "faiss_vector_count" in info:
                print(
                    f"  FAISS Vector Count: {_colorize(_format_value(info.get('faiss_vector_count')), 'cyan')}"
                )

    except Exception as e:
        print(f"{_colorize('ERROR:', 'red')} {e}", file=sys.stderr)
        sys.exit(1)


def setup_index_parser(subparsers: Any) -> None:
    """Add index subcommands to the CLI parser."""
    index_parser = subparsers.add_parser(
        "index",
        help="Index management commands",
        description="Manage file-based indexes: view stats, etc.",
    )

    index_subparsers = index_parser.add_subparsers(
        dest="index_command", title="Index Commands"
    )

    # index stats
    stats_parser = index_subparsers.add_parser(
        "stats",
        help="Show index statistics",
        description="Display index metadata and properties.",
    )
    stats_parser.add_argument("index_dir", help="Path to the index directory.")
    stats_parser.add_argument("--json", action="store_true", help="Output as JSON")
    stats_parser.set_defaults(func=cmd_index_stats)

    # Default: show help when no subcommand given
    def _default_index_handler(args: argparse.Namespace) -> None:
        if not args.index_command:
            index_parser.print_help()
            raise SystemExit(1)

    index_parser.set_defaults(func=_default_index_handler)
