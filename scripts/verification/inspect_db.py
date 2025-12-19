#!/usr/bin/env python3
"""
Database Inspection CLI.
Unified tool for checking DB matching, schema, and liveliness.
"""
import argparse
import sys
from pathlib import Path

from sqlalchemy import create_engine, inspect, text

# Add backend/src to path
root_dir = Path(__file__).resolve().parents[2]
sys.path.append(str(root_dir / "backend" / "src"))

from cortex.config.loader import get_config  # noqa: E402


def check_connection(_args):
    """Basic connectivity and table check."""
    print("Locked & Loaded. Checking connection...")
    config = get_config()
    db_url = config.database.url

    # Mask password
    safe_url = db_url.split("@")[-1] if "@" in db_url else "..."
    print(f"URL suffix: ...@{safe_url}")

    try:
        engine = create_engine(db_url)
        with engine.connect() as conn:
            print("✅ Connection successful.")

            # Check basic tables
            insp = inspect(engine)
            tables = insp.get_table_names()
            required = ["conversations", "chunks", "ingest_jobs"]
            missing = [t for t in required if t not in tables]

            if missing:
                print(f"❌ Missing tables: {missing}")
            else:
                print(f"✅ Core tables present: {required}")

            # Check row counts
            for t in required:
                if t in tables:
                    count = conn.execute(text(f"SELECT count(*) FROM {t}")).scalar()
                    print(f"   - {t}: {count} rows")

    except Exception as e:
        print(f"❌ Connection failed: {e}")
        sys.exit(1)


def list_databases(_args):
    """List all databases on the server (if permissions allow)."""
    config = get_config()
    db_url = config.database.url
    # Force defaultdb to list others if possible
    base_url = db_url.rsplit("/", 1)[0] + "/defaultdb"

    try:
        engine = create_engine(base_url)
        with engine.connect() as conn:
            print("\nDatabase List:")
            sql = text(
                "SELECT datname, pg_size_pretty(pg_database_size(datname)) as size FROM pg_database WHERE datistemplate = false;"
            )
            rows = conn.execute(sql).fetchall()
            for r in rows:
                print(f"- {r[0]} ({r[1]})")
    except Exception as e:
        print(f"Could not list databases (likely permission/connection issue): {e}")


def check_vector(_args):
    """Deep check of vector setup on 'chunks' table."""
    config = get_config()
    engine = create_engine(config.database.url)

    print("\nChecking Vector Extension & Columns...")
    try:
        with engine.connect() as conn:
            # Check extensions
            exts = conn.execute(text("SELECT extname FROM pg_extension")).fetchall()
            ext_names = [r[0] for r in exts]
            print(f"Extensions: {ext_names}")

            if "vector" not in ext_names:
                print("❌ 'vector' extension MISSING.")
            else:
                print("✅ 'vector' extension installed.")

            # Check column type
            insp = inspect(engine)
            cols = insp.get_columns("chunks")
            emb = next((c for c in cols if c["name"] == "embedding"), None)

            if emb:
                print(
                    f"✅ 'chunks.embedding' column found. Type reported: {emb['type']}"
                )

                # Deeper check
                sql = text(
                    """
                    SELECT format_type(atttypid, atttypmod)
                    FROM pg_attribute
                    WHERE attrelid = 'chunks'::regclass AND attname = 'embedding'
                """
                )
                real_type = conn.execute(sql).scalar()
                print(f"   Real Postgres Type: {real_type}")
            else:
                print("❌ 'chunks.embedding' column MISSING.")

    except Exception as e:
        print(f"Vector check failed: {e}")


def main():
    parser = argparse.ArgumentParser(description="Database Inspection Tool")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("check", help="Basic connection & table check")
    subparsers.add_parser("list", help="List all databases")
    subparsers.add_parser("vector", help="Deep check of vector columns")

    args = parser.parse_args()

    if args.command == "check":
        check_connection(args)
    elif args.command == "list":
        list_databases(args)
    elif args.command == "vector":
        check_vector(args)


if __name__ == "__main__":
    main()
