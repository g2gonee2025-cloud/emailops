"""
Maintenance commands.
"""

import argparse
import sys
from typing import Any, Protocol

from cortex_cli.style import colorize


class _Subparsers(Protocol):
    def add_parser(self, *args: Any, **kwargs: Any) -> argparse.ArgumentParser: ...


def resolve_entities(args: argparse.Namespace) -> None:
    """Run entity resolution."""
    dry_run = getattr(args, "dry_run", False)
    try:
        from cortex_workers.maintenance.resolve_entities import EntityResolver
    except ImportError as exc:
        print(
            f"{colorize('ERROR:', 'red')} Could not load resolution worker: {exc}",
            file=sys.stderr,
        )
        print(
            "Ensure the workers package is installed.",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        print(f"{colorize('▶ ENTITY RESOLUTION', 'bold')}")
        if dry_run:
            print(f"{colorize('DRY RUN', 'yellow')}")
        print()

        resolver = EntityResolver()
        resolver.run(dry_run=dry_run)

        print(f"\n{colorize('✓', 'green')} Entity resolution complete.")

    except Exception as e:
        import traceback

        traceback.print_exc(file=sys.stderr)
        print(f"{colorize('ERROR:', 'red')} {e}", file=sys.stderr)
        sys.exit(1)


def setup_maintenance_parser(
    subparsers: _Subparsers,
) -> None:
    """Setup maintenance command parser."""
    maintenance_parser = subparsers.add_parser(
        "maintenance",
        help="System maintenance tasks",
        description="Run maintenance tasks like entity resolution.",
    )
    maintenance_subparsers = maintenance_parser.add_subparsers(
        dest="maintenance_command",
        title="Maintenance Commands",
    )

    # resolve-entities
    resolve_parser = maintenance_subparsers.add_parser(
        "resolve-entities",
        help="Resolve duplicate entities in the graph",
    )
    resolve_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate resolution without making changes",
    )
    resolve_parser.set_defaults(func=resolve_entities)

    # Default: show available subcommands when no subcommand given
    def _default_maintenance_handler(args: argparse.Namespace | None = None) -> None:
        maintenance_command = getattr(args, "maintenance_command", None)
        if maintenance_command is None:
            print(f"{colorize('MAINTENANCE COMMANDS', 'bold')}\n")
            print("  resolve-entities  Resolve duplicate entities in the graph\n")
            print("Usage: cortex maintenance <command> [options]")
            raise SystemExit(1)

    maintenance_parser.set_defaults(func=_default_maintenance_handler)
