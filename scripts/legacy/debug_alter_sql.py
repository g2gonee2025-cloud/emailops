from cortex.config.loader import get_config
from sqlalchemy import create_engine, text


def raw_alter():
    try:
        config = get_config()
        # Use a fresh engine without pgvector types registered if possible, or just raw psycopg2
        # But let's use sqlalchemy text()
        engine = create_engine(config.database.url)

        with engine.connect() as conn:
            print("Attempting to create extension vector if not exists...")
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            conn.commit()

            print("Attempting ALTER COLUMN to halfvec...")
            # We use a transaction
            with conn.begin():
                conn.execute(
                    text(
                        "ALTER TABLE chunks ALTER COLUMN embedding TYPE halfvec(3840) USING embedding::halfvec(3840)"
                    )
                )
            print("Alter success.")

            print("Attempting Create Index...")
            with conn.begin():
                conn.execute(
                    text(
                        "CREATE INDEX IF NOT EXISTS ix_chunks_embedding ON chunks USING hnsw (embedding halfvec_cosine_ops) WITH (m = 16, ef_construction = 64)"
                    )
                )
            print("Index success.")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    raw_alter()
