import os
import sys
from pathlib import Path

# Add backend/src to sys.path
backend_src = Path("/root/workspace/emailops-vertex-ai/backend/src").resolve()
sys.path.append(str(backend_src))

try:
    from cortex.db.session import engine, get_db_session
    from sqlalchemy import inspect, text

    print("Testing DB connection...")
    with get_db_session() as session:
        result = session.execute(text("SELECT 1")).scalar()
        print(
            f"DB Selection Test: {'PASSED' if result == 1 else 'FAILED'} (Result: {result})"
        )

        # Check tables
        insp = inspect(engine)
        tables = insp.get_table_names()
        print(f"Tables found: {tables}")

        # CRUD Test: Insert a temporary audit log entry
        print("Running CRUD test...")
        import uuid
        from datetime import datetime

        # Using raw SQL to avoid model dependency issues in this smoke test if they exist
        tenant_id = "test-tenant"
        log_id = uuid.uuid4()
        session.execute(
            text(
                """
            INSERT INTO audit_log (audit_id, ts, tenant_id, user_or_agent, action, risk_level)
            VALUES (:id, :ts, :tid, :user, :action, :risk)
        """
            ),
            {
                "id": log_id,
                "ts": datetime.utcnow(),
                "tid": tenant_id,
                "user": "smoke-test-agent",
                "action": "SMOKE_TEST",
                "risk": "low",
            },
        )
        session.commit()
        print(f"Inserted audit log entry: {log_id}")

        # Read it back
        read_back = session.execute(
            text("SELECT action FROM audit_log WHERE audit_id = :id"), {"id": log_id}
        ).scalar()
        print(f"Read back action: {read_back}")

        # Delete it
        session.execute(
            text("DELETE FROM audit_log WHERE audit_id = :id"), {"id": log_id}
        )
        session.commit()
        print("Cleaned up smoke test data.")

    print("DB Smoke Test Completed Successfully.")
except Exception as e:
    print(f"DB Smoke Test FAILED: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)
