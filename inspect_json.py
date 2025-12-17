import os
import sys

from sqlalchemy import create_engine, text

sys.path.append(os.path.join(os.getcwd(), "backend", "src"))
from cortex.config.loader import get_config


def inspect_metadata():
    try:
        config = get_config()
        db_url = config.database.url
        if "sslmode" not in db_url:
            db_url += "?sslmode=require"

        engine = create_engine(db_url)
        with engine.connect() as conn:
            # Get samples from threads metadata where it's not empty/null
            sql = text(
                """
                SELECT metadata
                FROM threads
                WHERE metadata IS NOT NULL AND metadata::text != '{}'
                LIMIT 5
            """
            )
            result = conn.execute(sql).fetchall()

            print("----- Threads Metadata Samples -----")
            if not result:
                print("No non-empty metadata found in threads.")
            else:
                for row in result:
                    print(row.metadata)

            # Also check messages metadata as it might be richer
            print("\n----- Messages Metadata Samples -----")
            sql_msg = text(
                """
                SELECT metadata
                FROM messages
                WHERE metadata IS NOT NULL AND metadata::text != '{}'
                LIMIT 5
            """
            )
            result_msg = conn.execute(sql_msg).fetchall()

            if not result_msg:
                print("No non-empty metadata found in messages.")
            else:
                for row in result_msg:
                    print(row.metadata)

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    inspect_metadata()
