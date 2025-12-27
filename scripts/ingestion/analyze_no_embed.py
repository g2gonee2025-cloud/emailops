import os

import psycopg2

# --- Configuration ---
# Use environment variables for sensitive data like DB URLs
db_url = os.environ.get("DB_URL")
if not db_url:
    raise RuntimeError(
        "DB_URL environment variable is not set. Please set it to your Postgres connection URI."
    )

# Use a specific tenant_id for analysis
TENANT_ID = "deep-dive-tenant"


def analyze_tenant_data(tenant_id: str):
    """
    Analyzes and prints key statistics for a given tenant's data.

    Args:
        tenant_id: The ID of the tenant to analyze.
    """
    print(f"--- Analysis for Tenant: {tenant_id} ---")

    try:
        # --- Database Connection ---
        # Use 'with' statement for automatic connection closing
        with psycopg2.connect(db_url) as conn:
            # Use 'with' statement for automatic cursor closing
            with conn.cursor() as cur:
                # --- Query Execution ---
                # Helper function for executing and fetching single-value queries
                def fetch_count(query: str, params: tuple) -> int:
                    cur.execute(query, params)
                    result = cur.fetchone()
                    return result[0] if result else 0

                # 1. Threads
                thread_count = fetch_count(
                    "SELECT count(*) FROM threads WHERE tenant_id = %s", (tenant_id,)
                )
                print(f"Threads: {thread_count}")

                # 2. Messages
                msg_count = fetch_count(
                    "SELECT count(*) FROM messages WHERE tenant_id = %s", (tenant_id,)
                )
                print(f"Messages: {msg_count}")

                # 3. Attachments
                att_count = fetch_count(
                    "SELECT count(*) FROM attachments WHERE tenant_id = %s",
                    (tenant_id,),
                )
                print(f"Attachments: {att_count}")

                # 4. Attachments named 'attachments_log.csv' (Should be 0 if filtered)
                csv_count = fetch_count(
                    "SELECT count(*) FROM attachments WHERE tenant_id = %s AND filename = 'attachments_log.csv'",
                    (tenant_id,),
                )
                print(f"attachments_log.csv count: {csv_count} (Should be 0)")

                # 5. Chunks
                chunk_count = fetch_count(
                    "SELECT count(*) FROM chunks WHERE tenant_id = %s", (tenant_id,)
                )
                print(f"Total Chunks: {chunk_count}")

                # 6. Chunks with Embeddings (Should be 0)
                embed_count = fetch_count(
                    "SELECT count(*) FROM chunks WHERE tenant_id = %s AND embedding IS NOT NULL",
                    (tenant_id,),
                )
                print(f"Chunks with Embeddings: {embed_count} (Should be 0)")

                # 7. Chunks Stats
                cur.execute(
                    "SELECT AVG(LENGTH(text)), MAX(LENGTH(text)), AVG(char_end - char_start) FROM chunks WHERE tenant_id = %s",
                    (tenant_id,),
                )
                stats_result = cur.fetchone()
                if stats_result:
                    avg_len, max_len, avg_span = stats_result
                    print(
                        f"Chunk Text Lengths - Avg: {avg_len or 0:.1f}, Max: {max_len or 0}"
                    )
                    print(
                        f"Chunk Spans (char_end - char_start) - Avg: {avg_span or 0:.1f}"
                    )

    except psycopg2.Error as e:
        print(f"Database error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    analyze_tenant_data(TENANT_ID)
