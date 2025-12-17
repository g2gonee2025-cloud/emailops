import os
import sys

from sqlalchemy import create_engine, text

# Add backend/src to path
sys.path.append(os.path.join(os.getcwd(), "backend", "src"))
from cortex.config.loader import get_config


def audit_database():
    try:
        config = get_config()
        # Ensure SSL mode is set for DO managed DB
        db_url = config.database.url
        if "sslmode" not in db_url:
            db_url += "?sslmode=require"

        engine = create_engine(db_url)

        print(f"Connected to: {db_url.split('@')[-1]}")

        with engine.connect() as conn:
            # 1. List all tables
            print("\n----- Tables in 'public' schema -----")
            sql_tables = text(
                """
                SELECT
                    relname as table_name,
                    n_live_tup as row_count,
                    pg_size_pretty(pg_total_relation_size(relid)) as total_size
                FROM pg_stat_user_tables
                WHERE schemaname = 'public'
                ORDER BY n_live_tup DESC;
            """
            )
            result = conn.execute(sql_tables).fetchall()

            if not result:
                print("No tables found in public schema.")
            else:
                print(f"{'Table Name':<30} | {'Rows':<10} | {'Size':<10}")
                print("-" * 56)
                for row in result:
                    print(
                        f"{row.table_name:<30} | {row.row_count:<10} | {row.total_size:<10}"
                    )

            # 2. Check for NULL embeddings if chunks exists
            print("\n----- Data Integrity Checks -----")
            tables = [r.table_name for r in result]

            if "chunks" in tables:
                null_embeds = conn.execute(
                    text("SELECT count(*) FROM chunks WHERE embedding IS NULL")
                ).scalar()
                total_chunks = conn.execute(
                    text("SELECT count(*) FROM chunks")
                ).scalar()
                print(
                    f"Chunks: {total_chunks} total, {null_embeds} with NULL embeddings"
                )

                # Check for old dimensions if possible (heuristic)
                # We can't easily check vector dim per row in pure SQL without function,
                # but we can check if any fail to cast or just rely on the column type we saw earlier.
                # Actually, earlier we saw the column type is vector(3840), so all non-nulls MUST be 3840.

            if "alembic_version" in tables:
                rev = conn.execute(
                    text("SELECT version_num FROM alembic_version")
                ).scalar()
                print(f"Alembic Revision: {rev}")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    audit_database()
