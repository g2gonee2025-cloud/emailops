#!/usr/bin/env python3
"""
Validation script for S3 ingestion pipeline results.

Validates the results claimed in the walkthrough:
- 4,107 conversations processed
- 31,232 chunks created
- 8,253 attachments processed
- 0 failures
"""
import sys
from pathlib import Path

# Add backend/src to path
sys.path.insert(0, str(Path.cwd() / "backend" / "src"))

from cortex.config.loader import get_config
from sqlalchemy import create_engine, inspect, text


def validate_ingestion_results():
    print("=" * 60)
    print("  S3 INGESTION PIPELINE VALIDATION")
    print("=" * 60)

    # Load config and connect
    print("\n[1] Loading configuration...")
    try:
        config = get_config()
        db_url = config.database.url
        masked = db_url.split("@")[-1] if "@" in db_url else "..."
        print(f"    Database: ...@{masked}")
    except Exception as e:
        print(f"❌ Failed to load config: {e}")
        return False

    print("\n[2] Connecting to database...")
    try:
        engine = create_engine(db_url)
        inspector = inspect(engine)
    except Exception as e:
        print(f"❌ Failed to connect: {e}")
        return False

    # Expected values from walkthrough
    EXPECTED = {
        "conversations": 4107,
        "chunks": 31232,
        "attachments": 8253,
    }

    print("\n[3] Checking table existence...")
    existing_tables = inspector.get_table_names()
    required = ["conversations", "attachments", "chunks"]
    all_tables_ok = True
    for table in required:
        if table in existing_tables:
            print(f"    ✅ Table '{table}' exists")
        else:
            print(f"    ❌ Table '{table}' MISSING")
            all_tables_ok = False

    if not all_tables_ok:
        print("\n❌ Critical tables missing! Cannot validate.")
        return False

    print("\n[4] Validating row counts...")
    results = {}
    with engine.connect() as conn:
        for table in required:
            count = conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()
            results[table] = count
            expected = EXPECTED[table]
            diff = count - expected
            diff_pct = (diff / expected * 100) if expected > 0 else 0

            if count == expected:
                print(f"    ✅ {table}: {count:,} (exact match)")
            elif abs(diff_pct) < 1:  # Within 1%
                print(
                    f"    ⚠️  {table}: {count:,} (expected {expected:,}, diff: {diff:+,})"
                )
            else:
                print(
                    f"    ❌ {table}: {count:,} (expected {expected:,}, diff: {diff:+,})"
                )

    print("\n[5] Checking schema columns...")
    with engine.connect() as conn:
        # Check Chunk embedding dimension
        emb_type = conn.execute(
            text(
                """
            SELECT format_type(atttypid, atttypmod)
            FROM pg_attribute
            WHERE attrelid = 'chunks'::regclass
            AND attname = 'embedding'
        """
            )
        ).scalar()

        if emb_type and "3840" in emb_type:
            print(f"    ✅ chunks.embedding: {emb_type}")
        else:
            print(f"    ⚠️  chunks.embedding: {emb_type} (expected vector(3840))")

        # Check key columns exist
        chunk_cols = [c["name"] for c in inspector.get_columns("chunks")]
        required_cols = ["char_start", "char_end", "section_path", "is_attachment"]
        for col in required_cols:
            if col in chunk_cols:
                print(f"    ✅ chunks.{col} exists")
            else:
                print(f"    ❌ chunks.{col} MISSING")

    print("\n[6] Checking embeddings status...")
    with engine.connect() as conn:
        null_embeddings = conn.execute(
            text("SELECT COUNT(*) FROM chunks WHERE embedding IS NULL")
        ).scalar()
        non_null = conn.execute(
            text("SELECT COUNT(*) FROM chunks WHERE embedding IS NOT NULL")
        ).scalar()

        print(f"    Embeddings NULL:     {null_embeddings:,}")
        print(f"    Embeddings present:  {non_null:,}")

        if null_embeddings == results.get("chunks", 0):
            print(
                "    ✅ All chunks have NULL embeddings (as expected - no embedding generation)"
            )
        elif non_null > 0:
            print(f"    Info:  {non_null:,} chunks have embeddings")

    print("\n[7] Checking chunk size distribution...")
    with engine.connect() as conn:
        stats = conn.execute(
            text(
                """
            SELECT
                MIN(LENGTH(text)) as min_len,
                MAX(LENGTH(text)) as max_len,
                AVG(LENGTH(text))::int as avg_len,
                PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY LENGTH(text))::int as median_len
            FROM chunks
        """
            )
        ).fetchone()

        if stats:
            print("    Text length stats:")
            print(f"      Min:    {stats[0]:,} chars")
            print(f"      Max:    {stats[1]:,} chars")
            print(f"      Avg:    {stats[2]:,} chars")
            print(f"      Median: {stats[3]:,} chars")

    print("\n[8] Sampling data quality...")
    with engine.connect() as conn:
        # Sample a few conversations
        sample = conn.execute(
            text(
                """
            SELECT
                folder_name,
                subject,
                (SELECT COUNT(*) FROM chunks c WHERE c.conversation_id = conv.conversation_id) as chunk_count,
                (SELECT COUNT(*) FROM attachments a WHERE a.conversation_id = conv.conversation_id) as att_count
            FROM conversations conv
            ORDER BY RANDOM()
            LIMIT 5
        """
            )
        ).fetchall()

        print("    Sample conversations:")
        for row in sample:
            print(f"      📁 {row[0][:40]}...")
            print(f"         Subject: {(row[1] or 'N/A')[:50]}")
            print(f"         Chunks: {row[2]}, Attachments: {row[3]}")

    # Summary
    print("\n" + "=" * 60)
    print("  VALIDATION SUMMARY")
    print("=" * 60)

    total_expected = sum(EXPECTED.values())
    total_actual = sum(results.values())
    match_pct = (total_actual / total_expected * 100) if total_expected > 0 else 0

    print(f"\n  Expected total records: {total_expected:,}")
    print(f"  Actual total records:   {total_actual:,}")
    print(f"  Match percentage:       {match_pct:.1f}%")

    if total_actual == total_expected:
        print("\n  ✅ ALL VALIDATION CHECKS PASSED")
        return True
    elif match_pct >= 99:
        print("\n  ⚠️  VALIDATION PASSED (within 1% tolerance)")
        return True
    else:
        print("\n  ❌ VALIDATION FAILED - significant discrepancy")
        return False


if __name__ == "__main__":
    success = validate_ingestion_results()
    sys.exit(0 if success else 1)
