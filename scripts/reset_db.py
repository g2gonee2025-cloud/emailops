import os
import sys
import psycopg2
import traceback

def reset_db(db_url, db_admin_role):
    """
    Connects to the database, drops the public schema, and recreates it.
    """
    print("Connecting to database...")
    try:
        with psycopg2.connect(db_url) as conn:
            conn.autocommit = True
            with conn.cursor() as cur:
                # Drop public schema and recreate it
                print("Dropping public schema...")
                cur.execute("DROP SCHEMA public CASCADE;")
                cur.execute("CREATE SCHEMA public;")
                cur.execute(f"GRANT ALL ON SCHEMA public TO {db_admin_role};")
                cur.execute("REVOKE ALL ON SCHEMA public FROM public;")

                # Verify
                cur.execute(
                    "SELECT count(*) FROM information_schema.tables WHERE table_schema = 'public';"
                )
                count = cur.fetchone()[0]
                print(f"Public schema reset. Table count: {count}")

    except psycopg2.Error as e:
        print("An error occurred with the database operation.", file=sys.stderr)
        # Log the full traceback to a file for debugging, avoiding exposure of sensitive info
        with open("reset_db_error.log", "a") as f:
            f.write(f"Timestamp: {traceback.format_exc()}\n")
        print("Details have been logged to reset_db_error.log.", file=sys.stderr)
        sys.exit(1)


def main():
    """
    Main function to orchestrate the database reset process.
    """
    db_url = os.environ.get("OUTLOOKCORTEX_DB_URL")
    if not db_url:
        print(
            "Error: OUTLOOKCORTEX_DB_URL environment variable must be set for database operations",
            file=sys.stderr,
        )
        sys.exit(1)

    if "prod" in db_url.lower() or "production" in db_url.lower():
        print("Error: This script cannot be run on a production database.", file=sys.stderr)
        sys.exit(1)

    confirm = input(
        "Are you sure you want to drop the public schema? This is a destructive action. Type 'yes' to confirm: "
    )
    if confirm.lower() != "yes":
        print("Database reset cancelled.")
        sys.exit(0)

    db_admin_role = os.environ.get("DB_ADMIN_ROLE", "doadmin")
    reset_db(db_url, db_admin_role)


if __name__ == "__main__":
    main()
