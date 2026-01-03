#!/usr/bin/env python3
"""Checks database connectivity and migration status."""

import sys
from pathlib import Path


def main():
    """Main function to run the database check."""
    # Move imports inside main to catch ImportError gracefully
    try:
        # The script is in 'scripts/check', so project_root is three levels up.
        project_root = Path(__file__).resolve().parents[2]
        src_path = project_root / "backend" / "src"
        if str(src_path) not in sys.path:
            sys.path.insert(0, str(src_path))

        from cortex.config.loader import get_config
        from cortex.common.exceptions import ConfigurationError
        from sqlalchemy import create_engine, text
        from sqlalchemy.exc import SQLAlchemyError
    except ImportError as e:
        print(f"Error: Failed to import a required module: {type(e).__name__}", file=sys.stderr)
        print(f"Detail: {e}", file=sys.stderr)
        print("Please ensure you have installed the project dependencies from 'requirements.txt'.", file=sys.stderr)
        sys.exit(1)

    # Validate configuration
    try:
        config = get_config()
        if not config or not hasattr(config, "database") or not getattr(config.database, "url", None):
            print(
                "Error: Database URL is not configured. Please check your .env file or environment variables.",
                file=sys.stderr,
            )
            sys.exit(1)
        db_url = config.database.url
    except ConfigurationError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


    print("Attempting to connect to the database...")

    try:
        engine = create_engine(db_url)
        with engine.connect() as conn:
            # 1. Check basic connectivity
            conn.execute(text("SELECT 1")).scalar()
            print("Connectivity Check: SUCCESS")

            # 2. Check for migrations table
            try:
                version = conn.execute(text("SELECT version_num FROM alembic_version")).scalar()
                print(f"Migration Version Check: SUCCESS (Current Version: {version})")
            except SQLAlchemyError as e:
                # Specific error for migration check
                print("Migration Version Check: FAILED", file=sys.stderr)
                print(f"Error Type: {type(e).__name__}", file=sys.stderr)
                print("Could not retrieve migration version. The 'alembic_version' table may not exist or be accessible.", file=sys.stderr)
                sys.exit(1)

    except SQLAlchemyError as e:
        # Broad error for initial connection
        print("Connectivity Check: FAILED", file=sys.stderr)
        print(f"Error Type: {type(e).__name__}", file=sys.stderr)
        print("Could not connect to the database. Verify the URL, credentials, and network access.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        # Catch any other unexpected errors
        print("An unexpected error occurred: FAILED", file=sys.stderr)
        print(f"Error Type: {type(e).__name__}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
