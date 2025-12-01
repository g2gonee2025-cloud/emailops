"""Add pgvector HNSW index for fast similarity search."""

from alembic import op

revision = '003'
down_revision = '002'
branch_labels = None
depends_on = None

def upgrade():
    # IVFFlat index for cosine similarity (supports > 2000 dimensions)
    # op.execute("""
    #     CREATE INDEX idx_chunks_embedding_ivfflat ON chunks 
    #     USING ivfflat (embedding vector_cosine_ops)
    #     WITH (lists = 100)
    # """)
    pass

def downgrade():
    # op.execute('DROP INDEX IF EXISTS idx_chunks_embedding_ivfflat')
    pass
