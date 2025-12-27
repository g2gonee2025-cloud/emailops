"""Add graph entity tables for Knowledge Graph RAG.

Revision ID: 013_add_graph_entities
Revises: 012_update_audit_log
Create Date: 2024-12-24
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB, UUID

# revision identifiers
revision = "013_add_graph_entities"
down_revision = "012_update_audit_log"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create entity_nodes table
    op.create_table(
        "entity_nodes",
        sa.Column("node_id", UUID(as_uuid=True), primary_key=True),
        sa.Column("tenant_id", sa.String(64), nullable=False, index=True),
        sa.Column("name", sa.String(512), nullable=False),
        sa.Column("type", sa.String(64), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("properties", JSONB(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )

    # Create unique index for tenant_id + name
    op.create_index(
        "ix_entity_nodes_tenant_name",
        "entity_nodes",
        ["tenant_id", "name"],
        unique=True,
    )

    # Create entity_edges table
    op.create_table(
        "entity_edges",
        sa.Column("edge_id", UUID(as_uuid=True), primary_key=True),
        sa.Column("tenant_id", sa.String(64), nullable=False, index=True),
        sa.Column(
            "source_id",
            UUID(as_uuid=True),
            sa.ForeignKey("entity_nodes.node_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "target_id",
            UUID(as_uuid=True),
            sa.ForeignKey("entity_nodes.node_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("relation", sa.String(128), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("weight", sa.Float(), nullable=False, default=1.0),
        sa.Column(
            "conversation_id",
            UUID(as_uuid=True),
            sa.ForeignKey("conversations.conversation_id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )

    # Create indexes for edges
    op.create_index("ix_entity_edges_source", "entity_edges", ["source_id"])
    op.create_index("ix_entity_edges_target", "entity_edges", ["target_id"])
    op.create_index("ix_entity_edges_conversation", "entity_edges", ["conversation_id"])


def downgrade() -> None:
    op.drop_table("entity_edges")
    op.drop_table("entity_nodes")
