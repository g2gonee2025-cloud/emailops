"""Initial schema for EmailOps Cortex."""

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, TSVECTOR, UUID

revision = "001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    # Enable extensions
    op.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"')
    op.execute('CREATE EXTENSION IF NOT EXISTS "vector"')

    # Enums
    # op.execute("""
    #     CREATE TYPE chunk_type AS ENUM (
    #         'message_body', 'attachment_text', 'attachment_table',
    #         'quoted_history', 'other'
    #     )
    # """)
    # op.execute("CREATE TYPE attachment_status AS ENUM ('pending', 'parsed', 'unparsed_password_protected', 'failed')")
    # op.execute("CREATE TYPE risk_level AS ENUM ('low', 'medium', 'high')")

    # threads table
    op.create_table(
        "threads",
        sa.Column(
            "thread_id",
            UUID,
            primary_key=True,
            server_default=sa.text("uuid_generate_v4()"),
        ),
        sa.Column("tenant_id", sa.Text, nullable=False, index=True),
        sa.Column("subject_norm", sa.Text),
        sa.Column("original_subject", sa.Text),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("metadata", JSONB, server_default="{}"),
    )

    # messages table
    op.create_table(
        "messages",
        sa.Column("message_id", sa.Text, primary_key=True),
        sa.Column(
            "thread_id",
            UUID,
            sa.ForeignKey("threads.thread_id", ondelete="RESTRICT"),
            nullable=False,
            index=True,
        ),
        sa.Column("tenant_id", sa.Text, nullable=False, index=True),
        sa.Column("folder", sa.Text),
        sa.Column("sent_at", sa.DateTime(timezone=True)),
        sa.Column("recv_at", sa.DateTime(timezone=True)),
        sa.Column("from_addr", sa.Text, nullable=False),
        sa.Column("to_addrs", ARRAY(sa.Text)),
        sa.Column("cc_addrs", ARRAY(sa.Text)),
        sa.Column("bcc_addrs", ARRAY(sa.Text)),
        sa.Column("subject", sa.Text),
        sa.Column("body_plain", sa.Text),
        sa.Column("body_html", sa.Text),
        sa.Column("has_quoted_mask", sa.Boolean, server_default="false"),
        sa.Column("raw_storage_uri", sa.Text),
        sa.Column("tsv_subject_body", TSVECTOR),
        sa.Column("metadata", JSONB, server_default="{}"),
    )

    # attachments table
    op.create_table(
        "attachments",
        sa.Column(
            "attachment_id",
            UUID,
            primary_key=True,
            server_default=sa.text("uuid_generate_v4()"),
        ),
        sa.Column(
            "message_id",
            sa.Text,
            sa.ForeignKey("messages.message_id", ondelete="RESTRICT"),
            nullable=False,
            index=True,
        ),
        sa.Column("tenant_id", sa.Text, nullable=False, index=True),
        sa.Column("filename", sa.Text),
        sa.Column("mime_type", sa.Text),
        sa.Column("storage_uri_raw", sa.Text),
        sa.Column("storage_uri_extracted", sa.Text),
        sa.Column(
            "status",
            sa.Enum(
                "pending",
                "parsed",
                "unparsed_password_protected",
                "failed",
                name="attachment_status",
            ),
        ),
        sa.Column("extracted_chars", sa.Integer),
        sa.Column("metadata", JSONB, server_default="{}"),
    )

    # chunks table (with pgvector)
    op.create_table(
        "chunks",
        sa.Column(
            "chunk_id",
            UUID,
            primary_key=True,
            server_default=sa.text("uuid_generate_v4()"),
        ),
        sa.Column(
            "thread_id",
            UUID,
            sa.ForeignKey("threads.thread_id", ondelete="RESTRICT"),
            nullable=False,
            index=True,
        ),
        sa.Column(
            "message_id",
            sa.Text,
            sa.ForeignKey("messages.message_id", ondelete="RESTRICT"),
            index=True,
        ),
        sa.Column(
            "attachment_id",
            UUID,
            sa.ForeignKey("attachments.attachment_id", ondelete="RESTRICT"),
            index=True,
        ),
        sa.Column("tenant_id", sa.Text, nullable=False, index=True),
        sa.Column(
            "chunk_type",
            sa.Enum(
                "message_body",
                "attachment_text",
                "attachment_table",
                "quoted_history",
                "other",
                name="chunk_type",
            ),
        ),
        sa.Column("text", sa.Text),
        sa.Column("summary", sa.Text),
        sa.Column("section_path", sa.Text),
        sa.Column("position", sa.Integer),
        sa.Column("char_start", sa.Integer),
        sa.Column("char_end", sa.Integer),
        sa.Column(
            "embedding", Vector(1536)
        ),  # Initial dim; see 005_update_embedding_dim.py for upgrade to 3072
        sa.Column("embedding_model", sa.Text),
        sa.Column("tsv_text", TSVECTOR),
        sa.Column("metadata", JSONB, server_default="{}"),
    )

    # audit_log table
    op.create_table(
        "audit_log",
        sa.Column(
            "audit_id",
            UUID,
            primary_key=True,
            server_default=sa.text("uuid_generate_v4()"),
        ),
        sa.Column("ts", sa.DateTime(timezone=True), nullable=False, index=True),
        sa.Column("tenant_id", sa.Text, nullable=False, index=True),
        sa.Column("user_or_agent", sa.Text, nullable=False),
        sa.Column("action", sa.Text, nullable=False),
        sa.Column("input_hash", sa.Text),
        sa.Column("output_hash", sa.Text),
        sa.Column("policy_decisions", JSONB),
        sa.Column("risk_level", sa.Enum("low", "medium", "high", name="risk_level")),
        sa.Column("metadata", JSONB, server_default="{}"),
    )


def downgrade():
    op.drop_table("audit_log")
    op.drop_table("chunks")
    op.drop_table("attachments")
    op.drop_table("messages")
    op.drop_table("threads")
    op.execute("DROP TYPE IF EXISTS risk_level")
    op.execute("DROP TYPE IF EXISTS attachment_status")
    op.execute("DROP TYPE IF EXISTS chunk_type")
