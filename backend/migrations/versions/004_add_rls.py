"""Add Row-Level Security (RLS) for multi-tenancy.

Notes
- This enforces tenant isolation at the database level using the session setting
  app.current_tenant.
- FORCE ROW LEVEL SECURITY is enabled so the table owner is also subject to RLS.
"""

from alembic import op

revision = "004"
down_revision = "003"
branch_labels = None
depends_on = None


def upgrade():
    tables = ["threads", "messages", "attachments", "chunks", "audit_log"]

    for table in tables:
        op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")
        op.execute(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY")

        # Make the migration idempotent in case it is re-run in dev/test
        op.execute(f"DROP POLICY IF EXISTS tenant_isolation_{table} ON {table}")

        op.execute(
            f"""
            CREATE POLICY tenant_isolation_{table} ON {table}
            FOR ALL
            USING (tenant_id = current_setting('app.current_tenant', true))
            WITH CHECK (tenant_id = current_setting('app.current_tenant', true))
            """
        )


def downgrade():
    tables = ["threads", "messages", "attachments", "chunks", "audit_log"]

    for table in tables:
        op.execute(f"DROP POLICY IF EXISTS tenant_isolation_{table} ON {table}")
        op.execute(f"ALTER TABLE {table} NO FORCE ROW LEVEL SECURITY")
        op.execute(f"ALTER TABLE {table} DISABLE ROW LEVEL SECURITY")
