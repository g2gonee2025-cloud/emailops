import logging
import sys

from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

from cortex.config.loader import get_config

# --- Configuration Constants ---
# Schema identifiers
TABLE_NAME = "chunks"
COLUMN_NAME = "embedding"
INDEX_NAME = f"ix_{TABLE_NAME}_{COLUMN_NAME}"

# Vector and index parameters
# OpenAI text-embedding-3-large default dimension
VECTOR_DIMENSION = 3840
# HNSW graph degree (number of links per node)
HNSW_M = 16
# HNSW index build parameter (controls recall/latency)
HNSW_EF_CONSTRUCTION = 64

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)


def migrate_embedding_to_halfvec():
    """
    Performs a migration of the embedding column to halfvec type, creating the
    necessary extension and index within a single transaction.
    """
    try:
        config = get_config()
        db_url = getattr(getattr(config, "database", None), "url", None)
        if not db_url:
            logging.error("Invalid configuration: missing database.url")
            sys.exit(1)

        engine = create_engine(db_url)

        with engine.connect() as conn:
            # All operations are wrapped in a single transaction for atomicity.
            with conn.begin():
                logging.info("Ensuring 'vector' extension exists...")
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

                logging.warning(
                    "Starting ALTER TABLE... This is a blocking operation and may cause downtime on large tables."
                )
                alter_sql = text(
                    f"""
                    ALTER TABLE {TABLE_NAME}
                    ALTER COLUMN {COLUMN_NAME} TYPE halfvec(:dim)
                    USING {COLUMN_NAME}::halfvec(:dim)
                    """
                )
                conn.execute(alter_sql, {"dim": VECTOR_DIMENSION})
                logging.info("ALTER TABLE completed successfully.")

                logging.info(
                    "Starting CREATE INDEX CONCURRENTLY... This may take a long time but will not block writes."
                )
                index_sql = text(
                    f"""
                    CREATE INDEX CONCURRENTLY IF NOT EXISTS {INDEX_NAME}
                    ON {TABLE_NAME} USING hnsw ({COLUMN_NAME} halfvec_cosine_ops)
                    WITH (m = :m, ef_construction = :ef_construction)
                    """
                )
                conn.execute(
                    index_sql,
                    {"m": HNSW_M, "ef_construction": HNSW_EF_CONSTRUCTION},
                )
                logging.info("CREATE INDEX completed successfully.")

            logging.info("Migration completed successfully.")

    except SQLAlchemyError as e:
        logging.error(
            "A database error occurred during migration. "
            "The transaction has been rolled back. See logs for details."
        )
        # Log the full exception to stderr for debugging, but not to stdout.
        logging.debug(e, exc_info=True)
        sys.exit(1)
    except Exception as e:
        logging.error(f"An unexpected error occurred: {type(e).__name__}")
        logging.debug(e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    migrate_embedding_to_halfvec()
