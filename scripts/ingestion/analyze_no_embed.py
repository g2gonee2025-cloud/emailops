import argparse
import os
import sys

import psycopg2


def analyze_tenant_data(tenant_id: str, db_url: str):
    """
    Analyzes and prints key statistics for a given tenant's data.

    Args:
        tenant_id: The ID of the tenant to analyze.
        db_url: The PostgreSQL connection URI.
    """
    print(f"--- Analysis for Tenant: {tenant_id} ---")

    try:
        # --- Database Connection ---
        # Use 'with' statement for automatic connection closing
        with psycopg2.connect(db_url) as conn:
            # Use 'with' statement for automatic cursor closing
            with conn.cursor() as cur:
                # Helper function to execute and fetch a single row.
                def fetch_one(query: str, params: tuple):
                    cur.execute(query, params)
                    return cur.fetchone()

                # 1. Threads
                thread_count_result = fetch_one(
                    "SELECT count(*) FROM threads WHERE tenant_id = %s", (tenant_id,)
                )
                print(f"Threads: {thread_count_result[0] if thread_count_result else 0}")

                # 2. Messages
                msg_count_result = fetch_one(
                    "SELECT count(*) FROM messages WHERE tenant_id = %s", (tenant_id,)
                )
                print(f"Messages: {msg_count_result[0] if msg_count_result else 0}")

                # 3. Attachments
                attachments_query = """
                    SELECT
                        COUNT(*),
                        COUNT(*) FILTER (WHERE filename = 'attachments_log.csv')
                    FROM attachments
                    WHERE tenant_id = %s;
                """
                attachments_result = fetch_one(attachments_query, (tenant_id,))
                att_count, csv_count = attachments_result or (0, 0)
                print(f"Attachments: {att_count}")
                print(f"attachments_log.csv count: {csv_count} (Should be 0)")

                # 4. Chunks
                chunks_query = """
                    SELECT
                        COUNT(*),
                        COUNT(*) FILTER (WHERE embedding IS NOT NULL),
                        AVG(LENGTH(text)),
                        MAX(LENGTH(text)),
                        AVG(char_end - char_start)
                    FROM chunks
                    WHERE tenant_id = %s;
                """
                chunk_stats = fetch_one(chunks_query, (tenant_id,))

                # LOGIC_ERRORS: Check count before processing aggregates to avoid
                # misrepresenting NULLs as zeros.
                # The query always returns one row. Counts are 0 and aggregates are NULL
                # if there are no matching rows.
                chunk_count, embed_count, avg_len, max_len, avg_span = chunk_stats
                print(f"Total Chunks: {chunk_count}")
                print(f"Chunks with Embeddings: {embed_count} (Should be 0)")

                if chunk_count > 0:
                    print(
                        f"Chunk Text Lengths - Avg: {avg_len or 0:.1f}, Max: {max_len or 0}"
                    )
                    print(
                        f"Chunk Spans (char_end - char_start) - Avg: {avg_span or 0:.1f}"
                    )

    except psycopg2.Error as e:
        # SECURITY: Do not print raw exception, which could leak sensitive data.
        print(
            "Error: A database error occurred. Check connection details and permissions.",
            file=sys.stderr,
        )
        # EXCEPTION_HANDLING: Re-raise to provide a traceback and signal failure.
        raise
    except Exception as e:
        # SECURITY: Do not print raw exception.
        print("Error: An unexpected error occurred.", file=sys.stderr)
        # EXCEPTION_HANDLING: Re-raise to provide a traceback and signal failure.
        raise


def main():
    """Main function to parse arguments and run the analysis."""
    parser = argparse.ArgumentParser(
        description="Analyze tenant data in the database."
    )
    parser.add_argument(
        "--tenant-id",
        type=str,
        default="deep-dive-tenant",
        help="The tenant ID to analyze.",
    )
    parser.add_argument(
        "--db-url",
        type=str,
        default=os.environ.get("DB_URL"),
        help="The PostgreSQL connection URI. Defaults to DB_URL environment variable.",
    )
    args = parser.parse_args()

    if not args.db_url:
        print(
            "Error: Database URL is not provided. "
            "Set the DB_URL environment variable or use the --db-url flag.",
            file=sys.stderr,
        )
        sys.exit(1)

    analyze_tenant_data(args.tenant_id, args.db_url)


if __name__ == "__main__":
    main()
