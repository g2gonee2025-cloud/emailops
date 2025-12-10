import sys
from pathlib import Path

sys.path.append(str(Path("backend/src").resolve()))
from cortex.config.loader import get_config
from sqlalchemy import create_engine, text


def check_db():
    config = get_config()
    db_url = config.database.url

    print("Connecting to DB...")  # Masking URL in logs, but script sees it

    try:
        engine = create_engine(db_url)
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1")).scalar()
            print(f"Connectivity Check: SUCCESS (Result: {result})")

            # Check for migrations table
            result = conn.execute(
                text("SELECT version_num FROM alembic_version")
            ).scalar()
            print(f"Current Migration Version: {result}")

    except Exception as e:
        print(f"Connectivity Check: FAILED\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    check_db()
