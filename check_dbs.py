
import sys
import os
from sqlalchemy import create_engine, text

# Add backend/src to path
sys.path.append(os.path.join(os.getcwd(), 'backend', 'src'))
from cortex.config.loader import get_config

def list_databases():
    try:
        config = get_config()
        # Connect to defaultdb first to list others
        db_url = config.database.url
        if 'sslmode' not in db_url:
            db_url += '?sslmode=require'

        print(f"Connecting to: {db_url.split('@')[-1]}")
        engine = create_engine(db_url)

        with engine.connect() as conn:
            print("\n----- Databases on Server -----")
            sql = text("SELECT datname, pg_size_pretty(pg_database_size(datname)) as size FROM pg_database WHERE datistemplate = false;")
            result = conn.execute(sql).fetchall()

            for row in result:
                print(f"- {row.datname} ({row.size})")

            # Check if 'cortex' has tables if it exists
            cortex_exists = any(r.datname == 'cortex' for r in result)

        if cortex_exists:
            print("\n----- Checking 'cortex' database -----")
            # Switch connection to cortex
            cortex_url = db_url.replace('/defaultdb', '/cortex')
            cortex_engine = create_engine(cortex_url)
            try:
                with cortex_engine.connect() as conn:
                    result = conn.execute(text("SELECT count(*) FROM pg_stat_user_tables")).scalar()
                    print(f"Database 'cortex' has {result} tables.")
                    if result > 0:
                        print("(It seems you have data in BOTH defaultdb and cortex?)")
                    else:
                        print("(Database 'cortex' is empty of user tables.)")
            except Exception as e:
                print(f"Could not connect to cortex db: {e}")
        else:
            print("\nDatabase 'cortex' does NOT exist.")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    list_databases()
