"""Squashed schema (001-016).

Revision ID: 001_initial_schema
Revises: None
Create Date: 2025-12-31 00:00:00.000000

Includes:
- 001_initial_schema
- 010_add_fts_support
- 011_fix_vector_schema
- 012_update_audit_log_schema
- 013_add_graph_entities
- 014_add_summary_text_column
- 015_add_node_merge_func
- 015_add_summary_indexes
- 016_add_pagerank
"""

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import HALFVEC
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "001_initial_schema"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Extensions
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")

    # Conversations
    op.create_table(
        "conversations",
        sa.Column("conversation_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("tenant_id", sa.String(length=64), nullable=False),
        sa.Column("folder_name", sa.String(length=512), nullable=False),
        sa.Column("subject", sa.Text(), nullable=True),
        sa.Column("smart_subject", sa.Text(), nullable=True),
        sa.Column(
            "participants", postgresql.JSONB(astext_type=sa.Text()), nullable=True
        ),
        sa.Column("messages", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("earliest_date", sa.DateTime(timezone=True), nullable=True),
        sa.Column("latest_date", sa.DateTime(timezone=True), nullable=True),
        sa.Column("storage_uri", sa.Text(), nullable=True),
        sa.Column("extra_data", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column(
            "summary_text",
            sa.Text(),
            nullable=True,
            comment="AI-generated summary of the conversation",
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("conversation_id"),
    )
    op.create_index(
        "ix_conversations_tenant_folder",
        "conversations",
        ["tenant_id", "folder_name"],
        unique=False,
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_conversations_subject_fts ON conversations "
        "USING GIN (to_tsvector('english', coalesce(subject, '')))"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_conversations_summary_fts ON conversations "
        "USING GIN (to_tsvector('english', coalesce(summary_text, '')))"
    )

    # Attachments
    op.create_table(
        "attachments",
        sa.Column("attachment_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("conversation_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("filename", sa.String(length=512), nullable=True),
        sa.Column("content_type", sa.String(length=128), nullable=True),
        sa.Column("size_bytes", sa.Integer(), nullable=True),
        sa.Column("storage_uri", sa.Text(), nullable=True),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["conversation_id"],
            ["conversations.conversation_id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("attachment_id"),
    )
    op.create_index(
        "ix_attachments_conversation",
        "attachments",
        ["conversation_id"],
        unique=False,
    )

    # Chunks
    op.create_table(
        "chunks",
        sa.Column("chunk_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("tenant_id", sa.String(length=64), nullable=True),
        sa.Column("conversation_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("attachment_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("is_attachment", sa.Boolean(), nullable=False),
        sa.Column(
            "chunk_type",
            sa.String(length=32),
            server_default="message_body",
            nullable=True,
        ),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("tsv_text", postgresql.TSVECTOR(), nullable=True),
        sa.Column("embedding", HALFVEC(3840), nullable=True),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.Column("char_start", sa.Integer(), nullable=False),
        sa.Column("char_end", sa.Integer(), nullable=False),
        sa.Column("section_path", sa.String(length=256), nullable=True),
        sa.Column("extra_data", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["attachment_id"], ["attachments.attachment_id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["conversation_id"],
            ["conversations.conversation_id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("chunk_id"),
    )
    op.create_index("ix_chunks_conversation", "chunks", ["conversation_id"])
    op.create_index("ix_chunks_is_attachment", "chunks", ["is_attachment"])
    op.create_index("ix_chunks_tenant_id", "chunks", ["tenant_id"])
    op.create_index(
        "ix_chunks_tsv_text",
        "chunks",
        ["tsv_text"],
        postgresql_using="gin",
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_chunks_embedding "
        "ON chunks USING hnsw (embedding halfvec_cosine_ops) "
        "WITH (m = 16, ef_construction = 64);"
    )

    op.execute(
        """
        CREATE OR REPLACE FUNCTION chunks_tsv_trigger() RETURNS trigger AS $$
        BEGIN
            NEW.tsv_text := to_tsvector('english', COALESCE(NEW.text, ''));
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
    """
    )
    op.execute("DROP TRIGGER IF EXISTS trig_chunks_tsv ON chunks;")
    op.execute(
        """
        CREATE TRIGGER trig_chunks_tsv
            BEFORE INSERT OR UPDATE OF text ON chunks
            FOR EACH ROW
            EXECUTE FUNCTION chunks_tsv_trigger();
    """
    )

    # Audit Log
    op.create_table(
        "audit_log",
        sa.Column("audit_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column(
            "ts",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("tenant_id", sa.String(length=64), nullable=False),
        sa.Column("action", sa.String(length=128), nullable=False),
        sa.Column("user_or_agent", sa.String(length=256), nullable=True),
        sa.Column("input_hash", sa.String(length=64), nullable=True),
        sa.Column("output_hash", sa.String(length=64), nullable=True),
        sa.Column(
            "policy_decisions", postgresql.JSONB(astext_type=sa.Text()), nullable=True
        ),
        sa.Column(
            "risk_level", sa.String(length=32), server_default="low", nullable=False
        ),
        sa.Column("metadata_", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.PrimaryKeyConstraint("audit_id"),
    )
    op.create_index(op.f("ix_audit_log_tenant_id"), "audit_log", ["tenant_id"])
    op.create_index(op.f("ix_audit_log_timestamp"), "audit_log", ["ts"])

    # Entity Nodes
    op.create_table(
        "entity_nodes",
        sa.Column("node_id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("tenant_id", sa.String(64), nullable=False, index=True),
        sa.Column("name", sa.String(512), nullable=False),
        sa.Column("type", sa.String(64), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("properties", postgresql.JSONB(), nullable=True),
        sa.Column(
            "pagerank",
            sa.Float(),
            nullable=False,
            server_default="0.0",
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )
    op.create_index(
        "ix_entity_nodes_tenant_name",
        "entity_nodes",
        ["tenant_id", "name"],
        unique=True,
    )
    op.create_index(
        "ix_entity_nodes_pagerank",
        "entity_nodes",
        ["pagerank"],
        postgresql_ops={"pagerank": "DESC"},
    )
    op.create_index(
        "ix_entity_nodes_name_trgm",
        "entity_nodes",
        ["name"],
        postgresql_using="gin",
        postgresql_ops={"name": "gin_trgm_ops"},
    )

    # Entity Edges
    op.create_table(
        "entity_edges",
        sa.Column("edge_id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("tenant_id", sa.String(64), nullable=False, index=True),
        sa.Column(
            "source_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("entity_nodes.node_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "target_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("entity_nodes.node_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("relation", sa.String(128), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("weight", sa.Float(), nullable=False, default=1.0),
        sa.Column(
            "conversation_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("conversations.conversation_id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )
    op.create_index("ix_entity_edges_source", "entity_edges", ["source_id"])
    op.create_index("ix_entity_edges_target", "entity_edges", ["target_id"])
    op.create_index("ix_entity_edges_conversation", "entity_edges", ["conversation_id"])

    # Merge utility function
    op.execute(
        """
        CREATE OR REPLACE FUNCTION merge_entity_nodes(keep_id UUID, discard_id UUID)
        RETURNS VOID AS $$
        DECLARE
            discard_props JSONB;
        BEGIN
            SELECT properties INTO discard_props FROM entity_nodes WHERE node_id = discard_id;

            UPDATE entity_nodes
            SET properties = COALESCE(discard_props, '{}'::jsonb) || COALESCE(properties, '{}'::jsonb)
            WHERE node_id = keep_id;

            UPDATE entity_edges
            SET source_id = keep_id
            WHERE source_id = discard_id;

            UPDATE entity_edges
            SET target_id = keep_id
            WHERE target_id = discard_id;

            DELETE FROM entity_nodes WHERE node_id = discard_id;
        END;
        $$ LANGUAGE plpgsql;
    """
    )


def downgrade() -> None:
    op.execute("DROP FUNCTION IF EXISTS merge_entity_nodes(UUID, UUID);")

    op.drop_index("ix_entity_edges_conversation", table_name="entity_edges")
    op.drop_index("ix_entity_edges_target", table_name="entity_edges")
    op.drop_index("ix_entity_edges_source", table_name="entity_edges")
    op.drop_table("entity_edges")

    op.drop_index("ix_entity_nodes_name_trgm", table_name="entity_nodes")
    op.drop_index("ix_entity_nodes_pagerank", table_name="entity_nodes")
    op.drop_index("ix_entity_nodes_tenant_name", table_name="entity_nodes")
    op.drop_table("entity_nodes")

    op.drop_index(op.f("ix_audit_log_timestamp"), table_name="audit_log")
    op.drop_index(op.f("ix_audit_log_tenant_id"), table_name="audit_log")
    op.drop_table("audit_log")

    op.execute("DROP TRIGGER IF EXISTS trig_chunks_tsv ON chunks;")
    op.execute("DROP FUNCTION IF EXISTS chunks_tsv_trigger();")
    op.execute("DROP INDEX IF EXISTS ix_chunks_embedding;")
    op.drop_index("ix_chunks_tsv_text", table_name="chunks")
    op.drop_index("ix_chunks_tenant_id", table_name="chunks")
    op.drop_index("ix_chunks_is_attachment", table_name="chunks")
    op.drop_index("ix_chunks_conversation", table_name="chunks")
    op.drop_table("chunks")

    op.drop_index("ix_attachments_conversation", table_name="attachments")
    op.drop_table("attachments")

    op.execute("DROP INDEX IF EXISTS ix_conversations_summary_fts")
    op.execute("DROP INDEX IF EXISTS ix_conversations_subject_fts")
    op.drop_index("ix_conversations_tenant_folder", table_name="conversations")
    op.drop_table("conversations")
