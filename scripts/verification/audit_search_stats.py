import os
import sys
from pathlib import Path
from urllib.parse import urlparse, urlunparse

from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError


# Securely find the project root and add it to the path
def find_project_root(marker="pyproject.toml"):
    """Find the project root directory by searching for a marker file."""
    current_path = Path(__file__).resolve()
    while current_path != current_path.parent:
        if (current_path / marker).exists():
            return current_path
        current_path = current_path.parent
    raise FileNotFoundError(f"Project root with marker '{marker}' not found.")


# Add the backend/src to the system path
project_root = find_project_root()
sys.path.insert(0, str(project_root / "backend/src"))

from cortex.config.loader import get_config


def redact_db_url(db_url: str) -> str:
    """Redact password from the database URL for safe logging."""
    try:
        parsed = urlparse(db_url)
        if parsed.password:
            # Reconstruct the netloc without the password
            netloc = f"{parsed.username}:***@{parsed.hostname}"
            if parsed.port:
                netloc += f":{parsed.port}"
            # Rebuild the URL with the redacted netloc
            parsed = parsed._replace(netloc=netloc)
            return urlunparse(parsed)
        return db_url
    except Exception:
        # Fallback for unexpected URL formats
        return "Could not parse or redact DB URL"


def main():
    config = get_config()
    db_url = getattr(getattr(config, "database", None), "url", None)
    if not db_url:
        raise RuntimeError(
            "Database URL is not configured (config.database.url is None)."
        )

    redacted = redact_db_url(db_url)
    print(f"Connecting to DB: {redacted}")

    try:
        engine = create_engine(db_url)
        with engine.connect() as conn:
            print("--- Database Audit ---")

            # Consolidated query for performance
            try:
                summary_query = text(
                    """
                    SELECT
                        COUNT(*) AS total_chunks,
                        COUNT(CASE WHEN embedding IS NULL THEN 1 END) AS null_embeddings,
                        SUM(CASE WHEN text IS NULL THEN 1 ELSE 0 END) AS null_text,
                        SUM(CASE WHEN CHAR_LENGTH(text) < 50 THEN 1 ELSE 0 END) AS len_lt_50,
                        SUM(CASE WHEN CHAR_LENGTH(text) BETWEEN 50 AND 100 THEN 1 ELSE 0 END) AS len_50_100,
                        SUM(CASE WHEN CHAR_LENGTH(text) BETWEEN 100 AND 500 THEN 1 ELSE 0 END) AS len_100_500,
                        SUM(CASE WHEN CHAR_LENGTH(text) BETWEEN 500 AND 1000 THEN 1 ELSE 0 END) AS len_500_1000,
                        SUM(CASE WHEN CHAR_LENGTH(text) > 1000 THEN 1 ELSE 0 END) AS len_gt_1000
                    FROM chunks
                """
                )
                summary_result = conn.execute(summary_query).fetchone()

                if summary_result:
                    print(f"Total Chunks: {summary_result.total_chunks}")
                    print(f"Null Embeddings: {summary_result.null_embeddings}")

                    print("\nChunk Length Distribution:")
                    print(f"  NULL: {summary_result.null_text or 0}")
                    print(f"  < 50 chars: {summary_result.len_lt_50 or 0}")
                    print(f"  50-100 chars: {summary_result.len_50_100 or 0}")
                    print(f"  100-500 chars: {summary_result.len_100_500 or 0}")
                    print(f"  500-1000 chars: {summary_result.len_500_1000 or 0}")
                    print(f"  > 1000 chars: {summary_result.len_gt_1000 or 0}")

            except SQLAlchemyError as e:
                print(f"Error getting chunk summary: {e}")

            # 4. Attachment Stats
            try:
                print("\nAttachment Stats (is_attachment):")
                attach_stats_query = text(
                    """
                    SELECT is_attachment, COUNT(*)
                    FROM chunks
                    GROUP BY is_attachment
                """
                )
                attach_stats = conn.execute(attach_stats_query).fetchall()
                for is_att, cnt in attach_stats:
                    print(f"  {is_att}: {cnt}")
            except SQLAlchemyError as e:
                print(f"  Could not get attachment stats (column may be missing): {e}")

            # 5. Check if we have logs/csvs
            try:
                print("\nPotential Junk (files ending in .csv or .log):")
                junk_query = text(
                    """
                    SELECT COUNT(*)
                    FROM chunks
                    WHERE section_path LIKE '%.csv' OR section_path LIKE '%.log'
                """
                )
                junk = conn.execute(junk_query).scalar()
                print(f"  Chunks from .csv/.log files: {junk}")
            except SQLAlchemyError as e:
                print(f"  Could not get junk file stats (column may be missing): {e}")

    except SQLAlchemyError as e:
        print(f"Database connection failed: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
