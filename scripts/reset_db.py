import os

import psycopg2

# Load DB URL from environment; do not hardcode credentials
DB_URL = os.environ.get("OUTLOOKCORTEX_DB_URL")
if not DB_URL:
    raise RuntimeError(
        "OUTLOOKCORTEX_DB_URL environment variable must be set for database operations"
    )


def reset_db():
    print("Connecting to database...")
    try:
        conn = psycopg2.connect(DB_URL)
        conn.autocommit = True
        cur = conn.cursor()

        # Drop public schema and recreate it
        print("Dropping public schema...")
        cur.execute("DROP SCHEMA public CASCADE;")
        cur.execute("CREATE SCHEMA public;")
        cur.execute("GRANT ALL ON SCHEMA public TO doadmin;")
        cur.execute("REVOKE ALL ON SCHEMA public FROM public;")

        # Verify
        cur.execute(
            "SELECT count(*) FROM information_schema.tables WHERE table_schema = 'public';"
        )
        count = cur.fetchone()[0]
        print(f"Public schema reset. Table count: {count}")

        cur.close()
        conn.close()
    except Exception as e:
        print(f"Error resetting database: {e}")
        exit(1)


if __name__ == "__main__":
    reset_db()
