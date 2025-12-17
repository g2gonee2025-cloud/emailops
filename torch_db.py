import os
import sys

from sqlalchemy import create_engine, text

# Add backend/src to path
sys.path.append(os.path.join(os.getcwd(), "backend", "src"))
from cortex.config.loader import get_config


def torch():
    config = get_config()
    db_url = config.database.url
    if "sslmode" not in db_url:
        db_url += "?sslmode=require"

    print(f"TORCHING DATABASE: {db_url.split('@')[-1]}")
    engine = create_engine(db_url)

    with engine.connect() as conn:
        conn.execute(text("DROP SCHEMA public CASCADE;"))
        conn.execute(text("CREATE SCHEMA public;"))
        conn.execute(text("GRANT ALL ON SCHEMA public TO postgres;"))
        conn.execute(text("GRANT ALL ON SCHEMA public TO public;"))
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        conn.commit()

    print("Database wiped clean.")


if __name__ == "__main__":
    torch()
