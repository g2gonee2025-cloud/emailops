import os

import psycopg2

db_url = os.environ["DB_URL"]
conn = psycopg2.connect(db_url)
cur = conn.cursor()

# 1. Broad check: Look at all data from "deep-dive-tenant"
# (Since we used that tenant_id in verify_deep_dive.py)
tenant_id = "deep-dive-tenant"

print(f"--- Analysis for Tenant: {tenant_id} ---")

# Threads
cur.execute("SELECT count(*) FROM threads WHERE tenant_id = %s", (tenant_id,))
thread_count = cur.fetchone()[0]
print(f"Threads: {thread_count}")

# Messages
cur.execute("SELECT count(*) FROM messages WHERE tenant_id = %s", (tenant_id,))
msg_count = cur.fetchone()[0]
print(f"Messages: {msg_count}")

# Attachments
cur.execute("SELECT count(*) FROM attachments WHERE tenant_id = %s", (tenant_id,))
att_count = cur.fetchone()[0]
print(f"Attachments: {att_count}")

# Attachments named 'attachments_log.csv' (Should be 0 if filtered)
cur.execute(
    "SELECT count(*) FROM attachments WHERE tenant_id = %s AND filename = 'attachments_log.csv'",
    (tenant_id,),
)
csv_count = cur.fetchone()[0]
print(f"attachments_log.csv count: {csv_count} (Should be 0)")

# Chunks
cur.execute("SELECT count(*) FROM chunks WHERE tenant_id = %s", (tenant_id,))
chunk_count = cur.fetchone()[0]
print(f"Total Chunks: {chunk_count}")

# Chunks with Embeddings (Should be 0)
cur.execute(
    "SELECT count(*) FROM chunks WHERE tenant_id = %s AND embedding IS NOT NULL",
    (tenant_id,),
)
embed_count = cur.fetchone()[0]
print(f"Chunks with Embeddings: {embed_count} (Should be 0)")

# Chunks Stats
cur.execute(
    "SELECT AVG(LENGTH(text)), MAX(LENGTH(text)), AVG(char_end - char_start) FROM chunks WHERE tenant_id = %s",
    (tenant_id,),
)
avg_len, max_len, avg_span = cur.fetchone()
print(f"Chunk Text Lengths - Avg: {avg_len:.1f}, Max: {max_len}")
print(f"Chunk Spans (char_end - char_start) - Avg: {avg_span:.1f}")

conn.close()
