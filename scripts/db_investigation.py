import asyncio
import os
import sys
from datetime import UTC, datetime, timedelta

from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

# Ensure backend path is in sys.path
sys.path.append(os.path.abspath("backend/src"))

from cortex.config.loader import get_config
from cortex.db.models import (
    Attachment,
    AuditLog,
    Chunk,
    Conversation,
    EntityEdge,
)


async def analyze_db():
    config = get_config()
    db_url = config.database.url.replace("postgresql://", "postgresql+asyncpg://")
    db_url = db_url.replace("sslmode=require", "ssl=require")
    print(f"Connecting to: {db_url}")
    engine = create_async_engine(db_url)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with async_session() as session:
        print("\n--- Database Statistics ---")

        # 0. Subject Cleanup (Lunch/Booking)
        keyword_filters = [
            "%lunch%",
            "%booking%",
            "%out of office%",
            "%automatic reply%",
        ]

        # Count messages matching these keywords
        for keyword in keyword_filters:
            stmt = select(func.count(Conversation.conversation_id)).where(
                Conversation.subject.ilike(keyword)
            )
            count = (await session.execute(stmt)).scalar()
            print(f"Conversations matching '{keyword}': {count}")

        # 1. Audit Logs
        cutoff_30d = datetime.now(UTC) - timedelta(days=30)
        stmt = select(func.count(AuditLog.audit_id)).where(AuditLog.ts < cutoff_30d)
        old_audit_count = (await session.execute(stmt)).scalar()

        stmt_total = select(func.count(AuditLog.audit_id))
        total_audit_count = (await session.execute(stmt_total)).scalar()
        print(
            f"Audit Logs: {total_audit_count} total, {old_audit_count} older than 30 days"
        )

        # 2. Attachments
        cutoff_1h = datetime.now(UTC) - timedelta(hours=1)
        stmt = select(func.count(Attachment.attachment_id)).where(
            Attachment.status == "pending", Attachment.created_at < cutoff_1h
        )
        stuck_attachments = (await session.execute(stmt)).scalar()
        print(f"Stuck Attachments (>1h pending): {stuck_attachments}")

        # 3. Orphaned Edges
        stmt = select(func.count(EntityEdge.edge_id)).where(
            EntityEdge.conversation_id.is_(None)
        )
        orphaned_edges = (await session.execute(stmt)).scalar()
        print(f"Orphaned Edges (No Conversation): {orphaned_edges}")

        # 4. Orphaned Nodes (No edges)
        # Check nodes that have 0 incoming and 0 outgoing edges
        # This is more complex in SQL, skipping for simple check or doing left join
        # Simple check: Nodes with no edges
        stmt = text(
            """
            SELECT count(*) FROM entity_nodes n
            WHERE NOT EXISTS (SELECT 1 FROM entity_edges e WHERE e.source_id = n.node_id)
            AND NOT EXISTS (SELECT 1 FROM entity_edges e WHERE e.target_id = n.node_id)
        """
        )
        isolated_nodes = (await session.execute(stmt)).scalar()
        print(f"Isolated Entities (No Edges): {isolated_nodes}")

        # 5. NULL Embeddings
        stmt = select(func.count(Chunk.chunk_id)).where(Chunk.embedding.is_(None))
        null_embeddings = (await session.execute(stmt)).scalar()
        print(f"Chunks with NULL embeddings: {null_embeddings}")

    await engine.dispose()


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(analyze_db())
