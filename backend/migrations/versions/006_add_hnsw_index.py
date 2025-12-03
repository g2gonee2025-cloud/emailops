"""Add HNSW index for vector search.

Per pgvector official docs (v0.8.1+):
- HNSW provides better query performance than IVFFlat (speed-recall tradeoff)
- pgvector 0.5.0 bumped the limit to 16k dims, so we can index 3072-dim vectors
    directly without half-precision casts

Revision ID: 006
Revises: 005
Create Date: 2024-12-03
"""
from alembic import op

revision = "006"
down_revision = "005"
branch_labels = None
depends_on = None


def upgrade():
    """
    Create HNSW index directly on the 3072-dim embedding vectors.

    Index parameters:
    - m=16: default, good balance of recall vs index size
    - ef_construction=64: default, good balance of build time vs recall
    """
    # Drop any legacy index left over from earlier migrations so we do not
    # maintain two large HNSW structures simultaneously.
    op.execute("DROP INDEX IF EXISTS idx_chunks_embedding_hnsw")

    # Create HNSW index on chunks table without reducing dimensionality.
    op.execute(
        """
        CREATE INDEX CONCURRENTLY IF NOT EXISTS chunks_embedding_hnsw_idx 
        ON chunks 
        USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64)
    """
    )

    # Add GIN index on tsv_text for hybrid search (FTS component)
    op.execute(
        """
        CREATE INDEX CONCURRENTLY IF NOT EXISTS chunks_tsv_text_gin_idx
        ON chunks
        USING gin (tsv_text)
    """
    )

    # Add index on tenant_id for multi-tenant filtering
    op.execute(
        """
        CREATE INDEX CONCURRENTLY IF NOT EXISTS chunks_tenant_id_idx
        ON chunks (tenant_id)
    """
    )


def downgrade():
    op.execute("DROP INDEX IF EXISTS chunks_embedding_hnsw_idx")
    op.execute("DROP INDEX IF EXISTS chunks_tsv_text_gin_idx")
    op.execute("DROP INDEX IF EXISTS chunks_tenant_id_idx")

    # Recreate the legacy cosine index on the raw embedding column so that
    # earlier migrations (≤005) still have their original structure.
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hnsw ON chunks 
        USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64)
    """
    )
