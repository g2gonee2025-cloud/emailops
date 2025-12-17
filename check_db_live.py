import os

import sqlalchemy as sa
from sqlalchemy import inspect, text

# Get DB URL from environment
db_url = os.getenv("OUTLOOKCORTEX_DB_URL") or os.getenv("DB_URL")
if not db_url:
    print("❌ No DB_URL found in environment")
    exit(1)

print(f"Connecting to: {db_url.split('@')[-1]}...")  # Mask password
engine = sa.create_engine(db_url)
inspector = inspect(engine)

required_tables = [
    "threads",
    "messages",
    "attachments",
    "chunks",
    "ingest_jobs",
    "facts_ledger",
    "audit_log",
]

print("\n--- Table Check ---")
existing_tables = inspector.get_table_names()
all_tables_ok = True
for table in required_tables:
    if table in existing_tables:
        print(f"✅ Table '{table}' exists")
    else:
        print(f"❌ Table '{table}' MISSING")
        all_tables_ok = False

if not all_tables_ok:
    print("❌ Critical tables missing!")
    exit(1)

print("\n--- Column & Trigger Check ---")
with engine.connect() as conn:
    # Check embedding dimension
    # Note: parsing vector dimension from information_schema is tricky,
    # checking atdtypmod or pg_attribute is deeper.

    # Simple check: Does column exist?
    cols = [c["name"] for c in inspector.get_columns("chunks")]
    if "embedding" in cols:
        print("✅ Column 'chunks.embedding' exists")
    else:
        print("❌ Column 'chunks.embedding' MISSING")

    # Check for FTS trigger on messages
    result = conn.execute(
        text(
            """
        SELECT trigger_name
        FROM information_schema.triggers
        WHERE event_object_table = 'messages'
        AND trigger_name = 'tsvector_update_messages'
    """
        )
    ).scalar()

    if result:
        print(f"✅ Trigger '{result}' exists on messages")
    else:
        print("❌ Trigger 'tsvector_update_messages' MISSING")

    # Check for FTS trigger on chunks
    result = conn.execute(
        text(
            """
        SELECT trigger_name
        FROM information_schema.triggers
        WHERE event_object_table = 'chunks'
        AND trigger_name = 'tsvector_update_chunks'
    """
        )
    ).scalar()

    if result:
        print(f"✅ Trigger '{result}' exists on chunks")
    else:
        print("❌ Trigger 'tsvector_update_chunks' MISSING")

    # Check IngestJob table count
    count = conn.execute(text("SELECT count(*) FROM ingest_jobs")).scalar()
    print(f"\n✅ ingest_jobs table accessible (row count: {count})")

print("\n✨ Database Validation Complete ✨")
