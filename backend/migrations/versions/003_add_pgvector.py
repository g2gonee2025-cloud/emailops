"""Add pgvector HNSW index for fast similarity search.

With 1536 dimensions (compatible with DO pgvector's 2000 limit).
"""

from alembic import op

revision = "003"
down_revision = "002"
branch_labels = None
depends_on = None


def upgrade():
    # HNSW index for cosine similarity (faster than IVFFlat for < 1M vectors)
    op.execute(
        """
        CREATE INDEX idx_chunks_embedding_hnsw ON chunks 
        USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64)
    """
    )


def downgrade():
    op.execute("DROP INDEX IF EXISTS idx_chunks_embedding_hnsw")
