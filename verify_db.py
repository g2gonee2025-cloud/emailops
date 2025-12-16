
import sys
import os
from sqlalchemy import text, inspect

# Add backend/src to path so we can import cortex
sys.path.append(os.path.join(os.getcwd(), 'backend', 'src'))

from cortex.config.loader import get_config
from cortex.db.models import Base
from sqlalchemy import create_engine

def check_db():
    print("Loading configuration...")
    try:
        config = get_config()
        db_url = config.database.url
        print("Database URL found (masking password):")
        # Simple mask for display
        masked_url = db_url.split('@')[-1] if '@' in db_url else "..."
        print(f"  ...@{masked_url}")
    except Exception as e:
        print(f"Failed to load config: {e}")
        return

    print("\nConnecting to database...")
    try:
        engine = create_engine(db_url)
        with engine.connect() as conn:
            print("Connection successful!")

            # Inspect columns of 'chunks' table
            print("\nInspecting 'chunks' table...")
            insp = inspect(engine)
            columns = insp.get_columns('chunks')

            embedding_col = next((c for c in columns if c['name'] == 'embedding'), None)

            if embedding_col:
                print(f"Found 'embedding' column: {embedding_col['type']}")
                # calculated or type check
                # Verify if it is vector(3840)
                # The type object str usually looks like VECTOR(3840)
                print(f"Type string: {str(embedding_col['type'])}")
            else:
                print("ERROR: 'embedding' column NOT found in 'chunks' table.")

            # Run a raw query to specific check pg_attribute if needed, or just rely on inspector
            # Let's double check with raw SQL to be 100% sure of the dimension
            print("\nVerifying vector dimension via system catalog:")
            sql = text("""
                SELECT atttypmod
                FROM pg_attribute
                WHERE attrelid = 'chunks'::regclass
                AND attname = 'embedding';
            """)
            result = conn.execute(sql).scalar()
            # atttypmod for vector is often dim. But sometimes it needs decoding.
            # Easier: format_type

            sql2 = text("""
                SELECT format_type(atttypid, atttypmod)
                FROM pg_attribute
                WHERE attrelid = 'chunks'::regclass
                AND attname = 'embedding';
            """)
            result2 = conn.execute(sql2).scalar()
            print(f"Column Type (Postgres format): {result2}")

    except Exception as e:
        print(f"Database connection or inspection failed: {e}")

if __name__ == "__main__":
    check_db()
