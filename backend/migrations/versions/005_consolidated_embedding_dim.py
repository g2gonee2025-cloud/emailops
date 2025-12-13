"""Consolidated embedding dimension migration.

This migration replaces the following chained migrations:
- 005_update_embedding_dim.py (1536 -> 3072)
- 007_resize_embedding_dim_3840.py (3072 -> 3840)
- 008_resize_embedding_dim_1024.py (3840 -> 1024)
- 009_resize_embedding_dim_3840.py (1024 -> 3840)

Net effect: Sets embedding dimension to 3840 (for KaLM embeddings).

Revision ID: 005_consolidated_embedding_dim
Revises: ec7386d2401e
Create Date: 2025-12-12
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "005_consolidated_embedding_dim"
down_revision = "ec7386d2401e"
branch_labels = None
depends_on = None

# Target embedding dimension
TARGET_DIM = 3840
ORIGINAL_DIM = 1536

# SQLSTATE codes we care about
SQLSTATE_PROGRAM_LIMIT_EXCEEDED = "54000"
SQLSTATE_UNDEFINED_OBJECT = "42704"  # e.g. "access method hnsw does not exist"

HNSW_INDEX_NAME = "chunks_embedding_hnsw_idx"
IVFFLAT_INDEX_NAME = "chunks_embedding_ivfflat_idx"

HNSW_INDEX_SQL = f"""
    CREATE INDEX CONCURRENTLY IF NOT EXISTS {HNSW_INDEX_NAME}
    ON chunks
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64)
"""

IVFFLAT_INDEX_SQL = f"""
    CREATE INDEX CONCURRENTLY IF NOT EXISTS {IVFFLAT_INDEX_NAME}
    ON chunks
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 200)
"""

# Indexes that can depend on the embedding column or were created by earlier scripts
LEGACY_INDEXES = [
    "DROP INDEX IF EXISTS idx_chunks_embedding_hnsw",
    "DROP INDEX IF EXISTS ix_chunks_embedding",
    "DROP INDEX IF EXISTS chunks_embedding_hnsw_idx",  # older consolidated script name
    "DROP INDEX IF EXISTS chunks_embedding_ivfflat_idx",  # this script's ivfflat name
    "DROP INDEX IF EXISTS chunks_tsv_text_gin_idx",  # older consolidated script name
    "DROP INDEX IF EXISTS chunks_tenant_id_idx",  # older consolidated script name
]


def _sqlstate(exc: Exception) -> str | None:
    """Try to read a PostgreSQL SQLSTATE code in a driver-agnostic way."""
    orig = getattr(exc, "orig", exc)
    return getattr(orig, "sqlstate", None) or getattr(orig, "pgcode", None)


def _should_fallback_from_hnsw(exc: Exception) -> bool:
    """Return True for errors where we should fall back from HNSW to IVFFlat."""
    code = _sqlstate(exc)
    if code in (SQLSTATE_PROGRAM_LIMIT_EXCEEDED, SQLSTATE_UNDEFINED_OBJECT):
        return True

    msg = str(getattr(exc, "orig", exc)).lower()
    if "program limit exceeded" in msg:
        return True
    if "access method" in msg and "hnsw" in msg:
        return True

    return False


def _is_vector_dimension_mismatch(exc: Exception) -> bool:
    """Heuristic for pgvector dimension mismatch errors."""
    msg = str(getattr(exc, "orig", exc)).lower()
    return "vector" in msg and "dimension" in msg


def _drop_vector_indexes() -> None:
    op.execute(f"DROP INDEX IF EXISTS {HNSW_INDEX_NAME}")
    op.execute(f"DROP INDEX IF EXISTS {IVFFLAT_INDEX_NAME}")


def _has_index_on_column(conn, table_name: str, column_name: str) -> bool:
    """Detect any index that includes a given column (any index name)."""
    sql = sa.text(
        """
        SELECT 1
        FROM pg_index i
        JOIN pg_class t ON t.oid = i.indrelid
        JOIN pg_namespace n ON n.oid = t.relnamespace
        JOIN pg_attribute a ON a.attrelid = t.oid
        WHERE n.nspname = current_schema()
          AND t.relname = :table_name
          AND a.attname = :column_name
          AND a.attnum = ANY(i.indkey)
        LIMIT 1
        """
    )
    return (
        conn.execute(
            sql, {"table_name": table_name, "column_name": column_name}
        ).scalar()
        is not None
    )


def _recreate_vector_index() -> None:
    """Create vector index with HNSW, falling back to IVFFlat where needed."""
    _drop_vector_indexes()

    try:
        op.execute(HNSW_INDEX_SQL)
        return
    except Exception as exc:
        if not _should_fallback_from_hnsw(exc):
            raise

    # HNSW failed, try IVFFlat
    _drop_vector_indexes()
    try:
        op.execute(IVFFLAT_INDEX_SQL)
    except Exception as ivf_exc:
        if _sqlstate(ivf_exc) == SQLSTATE_PROGRAM_LIMIT_EXCEEDED:
            print(
                "WARNING: Vector index creation skipped (dimensions or settings exceed managed Postgres limits)."
            )
            return
        raise


def upgrade() -> None:
    """Upgrade embedding column to target dimension and rebuild indexes."""
    # Drop indexes that depend on embedding dimension
    for stmt in LEGACY_INDEXES:
        op.execute(stmt)

    # Resize the vector column to target dimension.
    #
    # If there is existing data with the old dimension, Postgres cannot cast it.
    # In that case, we fall back to setting existing values to NULL so the schema
    # can move forward and embeddings can be recomputed later.
    try:
        op.execute(
            f"ALTER TABLE chunks ALTER COLUMN embedding TYPE vector({TARGET_DIM})"
        )
    except Exception as exc:
        if not _is_vector_dimension_mismatch(exc):
            raise
        op.execute(
            f"ALTER TABLE chunks ALTER COLUMN embedding TYPE vector({TARGET_DIM}) "
            f"USING NULL::vector({TARGET_DIM})"
        )

    # Recreate vector index (plus any missing supporting indexes) without holding long locks
    ctx = op.get_context()
    with ctx.autocommit_block():
        _recreate_vector_index()

        # Avoid duplicate indexes: only create these if none already exist on the column.
        conn = op.get_bind()
        if not _has_index_on_column(conn, "chunks", "tsv_text"):
            op.execute(
                """
                CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_chunks_fts
                ON chunks
                USING gin (tsv_text)
                """
            )

        if not _has_index_on_column(conn, "chunks", "tenant_id"):
            op.execute(
                """
                CREATE INDEX CONCURRENTLY IF NOT EXISTS chunks_tenant_id_idx
                ON chunks (tenant_id)
                """
            )


def downgrade() -> None:
    """Revert to original embedding dimension."""
    # Drop indexes created by this migration (if present)
    op.execute(f"DROP INDEX IF EXISTS {HNSW_INDEX_NAME}")
    op.execute(f"DROP INDEX IF EXISTS {IVFFLAT_INDEX_NAME}")
    op.execute("DROP INDEX IF EXISTS chunks_tenant_id_idx")

    # Revert to original dimension (may require NULLing mismatched rows)
    try:
        op.execute(
            f"ALTER TABLE chunks ALTER COLUMN embedding TYPE vector({ORIGINAL_DIM})"
        )
    except Exception as exc:
        if not _is_vector_dimension_mismatch(exc):
            raise
        op.execute(
            f"ALTER TABLE chunks ALTER COLUMN embedding TYPE vector({ORIGINAL_DIM}) "
            f"USING NULL::vector({ORIGINAL_DIM})"
        )

    # Restore legacy vector index name from revision 003
    try:
        op.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hnsw ON chunks
            USING hnsw (embedding vector_cosine_ops)
            WITH (m = 16, ef_construction = 64)
            """
        )
    except Exception as exc:
        if not _should_fallback_from_hnsw(exc):
            raise

        op.execute("DROP INDEX IF EXISTS idx_chunks_embedding_hnsw")
        op.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hnsw ON chunks
            USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 200)
            """
        )
