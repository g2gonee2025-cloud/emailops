
import sys
import os
import logging
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# Add backend/src to path
sys.path.append(os.path.join(os.getcwd(), 'backend', 'src'))
from cortex.config.loader import get_config
from cortex.embeddings.client import get_embedding

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fix_embeddings")

def fix_embeddings():
    try:
        config = get_config()
        db_url = config.database.url
        if 'sslmode' not in db_url:
            db_url += '?sslmode=require'

        logger.info(f"Connecting to DB...")
        engine = create_engine(db_url)
        Session = sessionmaker(bind=engine)
        session = Session()

        # 1. Check for NULL embeddings
        check_sql = text("SELECT count(*) FROM chunks WHERE embedding IS NULL")
        count = session.execute(check_sql).scalar()
        logger.info(f"Found {count} chunks with NULL embeddings.")

        if count == 0:
            logger.info("Nothing to do.")
            return

        # 2. Fetch chunks
        # batch size 50
        offset = 0
        limit = 50

        total_fixed = 0

        while True:
            fetch_sql = text("""
                SELECT chunk_id, text
                FROM chunks
                WHERE embedding IS NULL
                LIMIT :limit
            """)

            rows = session.execute(fetch_sql, {"limit": limit}).fetchall()
            if not rows:
                break

            logger.info(f"Processing batch of {len(rows)}...")

            for row in rows:
                chunk_id = row.chunk_id
                chunk_text = row.text

                if not chunk_text:
                    logger.warning(f"Chunk {chunk_id} has empty text, skipping.")
                    continue

                try:
                    # Generate embedding
                    embedding = get_embedding(chunk_text)

                    # Update DB
                    update_sql = text("""
                        UPDATE chunks
                        SET embedding = :emb
                        WHERE chunk_id = :id
                    """)

                    # pgvector expects a list/array
                    session.execute(update_sql, {"emb": embedding, "id": chunk_id})
                    total_fixed += 1

                except Exception as e:
                    logger.error(f"Failed to embed chunk {chunk_id}: {e}")

            session.commit()
            logger.info(f"Committed batch. Total fixed: {total_fixed}")

        logger.info("Done!")

    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise

if __name__ == "__main__":
    fix_embeddings()
