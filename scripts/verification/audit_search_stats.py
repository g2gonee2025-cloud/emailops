import sys

sys.path.insert(0, "backend/src")
from cortex.config.loader import get_config
from sqlalchemy import create_engine, text


def main():
    config = get_config()
    db_url = getattr(getattr(config, "database", None), "url", None)
    if not db_url:
        raise RuntimeError(
            "Database URL is not configured (config.database.url is None)."
        )
    redacted = db_url.split("@")[-1] if "@" in db_url else db_url
    print(f"Connecting to DB: {redacted}")  # redact password
    engine = create_engine(db_url)

    with engine.connect() as conn:
        print("--- Database Audit ---")

        # 1. Total Chunks
        count = conn.execute(text("SELECT COUNT(*) FROM chunks")).scalar()
        print(f"Total Chunks: {count}")

        # 2. Null Embeddings
        nulls = conn.execute(
            text("SELECT COUNT(*) FROM chunks WHERE embedding IS NULL")
        ).scalar()
        print(f"Null Embeddings: {nulls}")

        # 3. Short Chunks Analysis
        print("\nChunk Length Distribution:")
        dist = conn.execute(
            text(
                """
            SELECT
                CASE
                    WHEN LENGTH(text) < 50 THEN '< 50 chars'
                    WHEN LENGTH(text) BETWEEN 50 AND 100 THEN '50-100 chars'
                    WHEN LENGTH(text) BETWEEN 100 AND 500 THEN '100-500 chars'
                    WHEN LENGTH(text) BETWEEN 500 AND 1000 THEN '500-1000 chars'
                    ELSE '> 1000 chars'
                END as bucket,
                COUNT(*) as cnt
            FROM chunks
            GROUP BY 1
            ORDER BY cnt DESC
        """
            )
        ).fetchall()
        for bucket, cnt in dist:
            print(f"  {bucket}: {cnt}")

        # 4. Attachment Stats
        try:
            print("\nAttachment Stats (is_attachment):")
            attach_stats = conn.execute(
                text(
                    """
                SELECT is_attachment, COUNT(*)
                FROM chunks
                GROUP BY is_attachment
            """
                )
            ).fetchall()
            for is_att, cnt in attach_stats:
                print(f"  {is_att}: {cnt}")
        except Exception as e:
            print(f"  (is_attachment column problem: {e})")

        # 5. Check if we have logs/csvs
        # assuming 'section_path' or similar exists from history?
        try:
            print("\nPotential Junk (files ending in .csv):")
            # Need to join with conversations? Or is metadata in chunks?
            # From history: "char_start, char_end, section_path fields in the 'Chunk' table"
            junk = conn.execute(
                text(
                    """
                SELECT COUNT(*)
                FROM chunks
                WHERE section_path LIKE '%.csv' OR section_path LIKE '%.log'
            """
                )
            ).scalar()
            print(f"  Chunks from .csv/.log files: {junk}")
        except Exception as e:
            print(f"  (section_path column problem: {e})")


if __name__ == "__main__":
    main()
