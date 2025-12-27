from cortex.config.loader import get_config
from sqlalchemy import create_engine, text


def raw_alter():
    try:
        config = get_config()
        # Use a fresh engine without pgvector types registered if possible, or just raw psycopg2
        # But let's use sqlalchemy text()
        db_url = getattr(getattr(config, "database", None), "url", None)
        if not db_url:
            raise RuntimeError("Invalid configuration: missing database.url")
        engine = create_engine(db_url)
        # The SQL executed below references the following values:
        # - 3840: vector embedding dimension for columns defined as vector(3840) (e.g., OpenAI text-embedding-3-large).
        # - m = 16: HNSW graph degree (number of links per node) used in index creation.
        # - ef_construction = 64: HNSW index build parameter controlling recall/latency during construction.
        # Keep these documented here to avoid "magic numbers" and adjust as needed for your deployment.

        with engine.connect() as conn:
            print("Attempting to create extension vector if not exists...")
            with conn.begin():
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

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
