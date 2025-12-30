"""Add pagerank column to entity_nodes for Graph RAG scoring.

Revision ID: 016_add_pagerank
Revises: 015_add_summary_indexes, 015_add_node_merge_func
Create Date: 2024-12-29
"""

import sqlalchemy as sa
from alembic import op

# revision identifiers
revision = "016_add_pagerank"
# Merge point: depend on both 015 heads
down_revision = ("015_add_summary_indexes", "015_add_node_merge_func")
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add pagerank column with default value
    op.add_column(
        "entity_nodes",
        sa.Column("pagerank", sa.Float(), nullable=False, server_default="0.0"),
    )

    # Create index for pagerank-based queries (high pagerank = important entities)
    op.create_index(
        "ix_entity_nodes_pagerank",
        "entity_nodes",
        ["pagerank"],
        postgresql_ops={"pagerank": "DESC"},
    )

    # Create trigram extension if not exists (for fuzzy entity name matching)
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm;")

    # Create trigram index for fuzzy entity name search
    op.create_index(
        "ix_entity_nodes_name_trgm",
        "entity_nodes",
        ["name"],
        postgresql_using="gin",
        postgresql_ops={"name": "gin_trgm_ops"},
    )


def downgrade() -> None:
    op.drop_index("ix_entity_nodes_name_trgm", table_name="entity_nodes")
    op.drop_index("ix_entity_nodes_pagerank", table_name="entity_nodes")
    op.drop_column("entity_nodes", "pagerank")
