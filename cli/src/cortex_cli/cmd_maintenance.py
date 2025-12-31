"""
Maintenance commands.
"""

import argparse
import sys

from cortex_cli.style import colorize


def resolve_entities(args: argparse.Namespace) -> None:
    """Run entity resolution."""
    try:
        from cortex_workers.maintenance.resolve_entities import EntityResolver

        print(f"{colorize('▶ ENTITY RESOLUTION', 'bold')}")
        if args.dry_run:
            print(f"{colorize('DRY RUN', 'yellow')}")
        print()

        resolver = EntityResolver()
        resolver.run(dry_run=getattr(args, "dry_run", False))

        print(f"\n{colorize('✓', 'green')} Entity resolution complete.")

    except ImportError:
        print(
            f"{colorize('ERROR:', 'red')} Could not load resolution worker. Ensure workers package is installed."
        )
        sys.exit(1)
    except Exception as e:
        import traceback

        traceback.print_exc()
        print(f"{colorize('ERROR:', 'red')} {e}")
        sys.exit(1)


def _run_maintenance_resolve(args: argparse.Namespace) -> None:
    """Wrapper for resolve-entities command."""
    resolve_entities(args)


def setup_maintenance_parser(
    subparsers: "argparse._SubParsersAction",
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
    def _default_maintenance_handler(args: argparse.Namespace | None = None) -> int:
        maintenance_command = getattr(args, "maintenance_command", None)
        if maintenance_command is None:
            print(f"{colorize('MAINTENANCE COMMANDS', 'bold')}\n")
            print("  resolve-entities  Resolve duplicate entities in the graph\n")
            print("Usage: cortex maintenance <command> [options]")
            return 1
        return 0
        # Ensure explicit return to avoid ambiguous control flow for callers
        return None
        # Ensure explicit return to avoid ambiguous control flow for callers
        return None

    maintenance_parser.set_defaults(func=_default_maintenance_handler)
