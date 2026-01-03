
import os
import sys
from urllib.parse import urlparse

import psycopg2
from dotenv import load_dotenv

def main():
    """Connects to the database and verifies the connection."""
    load_dotenv()

    db_url = os.getenv("OUTLOOKCORTEX_DB_URL")

    if not db_url:
        print("❌ ERROR: OUTLOOKCORTEX_DB_URL environment variable not set.", file=sys.stderr)
        sys.exit(1)

    try:
        # Safely log the connection target without credentials
        parsed_url = urlparse(db_url)
        safe_display_url = f"{parsed_url.hostname}:{parsed_url.port or 5432}/{parsed_url.path[1:]}"
        print(f"ℹ️  Connecting to: {safe_display_url}")

        with psycopg2.connect(db_url) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                result = cur.fetchone()
                print(f"✅ Query result: {result}")

    except psycopg2.Error:
        print("❌ Connection failed: Could not connect to the database.", file=sys.stderr)
        print("   Please check the OUTLOOKCORTEX_DB_URL and network connectivity.", file=sys.stderr)
        sys.exit(1)

    except Exception as e:
        print(f"❌ An unexpected error occurred: {type(e).__name__}", file=sys.stderr)
        sys.exit(1)

    print("✅ Database connection verified successfully.")

if __name__ == "__main__":
    main()
