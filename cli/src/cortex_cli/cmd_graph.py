"""CLI for graph-related commands."""

import argparse
import sys
from typing import Any

from cortex_cli.style import colorize as _colorize


def cmd_discover_schema(args: argparse.Namespace) -> None:
    """Discover the graph schema from a sample of conversations."""
    try:
        from cortex.intelligence.graph_discovery import (
            discover_graph_schema as discover_schema_logic,
        )

        tenant_id = getattr(args, "tenant_id", "default")
        sample_size = getattr(args, "sample_size", 20)

        print(f"\n{_colorize('GRAPH SCHEMA DISCOVERY', 'bold')}\n")
        print(f"  Tenant:      {_colorize(tenant_id, 'cyan')}")
        print(f"  Sample Size: {_colorize(str(sample_size), 'cyan')}")
        print(f"\n  {_colorize('⏳', 'yellow')} Discovering schema...")

        discover_schema_logic(tenant_id=tenant_id, sample_size=sample_size)

        print(f"\n  {_colorize('✓', 'green')} Schema discovery complete!")

    except ImportError as e:
        print(
            f"{_colorize('ERROR:', 'red')} Graph discovery module not available: {e}",
            file=sys.stderr,
        )
        sys.exit(1)
    except Exception as e:
        print(f"{_colorize('ERROR:', 'red')} {e}", file=sys.stderr)
        sys.exit(1)


def setup_graph_parser(subparsers: Any) -> None:
    """Add graph subcommands to the CLI parser."""
    graph_parser = subparsers.add_parser(
        "graph",
        help="Knowledge Graph commands",
        description="Commands for interacting with the Knowledge Graph.",
    )

    graph_subparsers = graph_parser.add_subparsers(
        dest="graph_command", title="Graph Commands"
    )

    # graph discover-schema
    discover_parser = graph_subparsers.add_parser(
        "discover-schema",
        help="Discover the graph schema from conversations",
        description="Analyze conversations to discover entity types and relationships.",
    )
    discover_parser.add_argument(
        "--tenant-id",
        "-t",
        default="default",
        help="Tenant ID to search within (default: default)",
    )
    discover_parser.add_argument(
        "--sample-size",
        "-n",
        type=int,
        default=20,
        help="Number of conversations to sample (default: 20)",
    )
    discover_parser.set_defaults(func=cmd_discover_schema)

    # Default: show help when no subcommand given
    def _default_graph_handler(args: argparse.Namespace) -> None:
        if not args.graph_command:
            graph_parser.print_help()

    graph_parser.set_defaults(func=_default_graph_handler)
