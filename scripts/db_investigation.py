import asyncio
import logging
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

from sqlalchemy import case, func, select
from sqlalchemy.engine.url import make_url
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

# --- Configuration & Constants ---

# Safely find the project root and add it to the path
try:
    _project_root = Path(__file__).resolve().parent.parent
    sys.path.append(str(_project_root / "backend" / "src"))
    from cortex.config.loader import get_config
    from cortex.db.models import (
        Attachment,
        AuditLog,
        Chunk,
        Conversation,
        EntityEdge,
        EntityNode,
    )
except ImportError as e:
    print(f"Error: Failed to set up Python path. Make sure the script is in the 'scripts' directory. Details: {e}")
    sys.exit(1)


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

THIRTY_DAYS = timedelta(days=30)
ONE_HOUR = timedelta(hours=1)


async def analyze_db():
    """Connects to the database and runs a series of analytical queries."""
    engine = None
    try:
        config = get_config()
        if not (config and config.database and config.database.url):
            logger.error("Database URL is not configured.")
            return

        db_url = make_url(config.database.url)

        # Robustly replace the synchronous driver with the asyncpg driver.
        if db_url.drivername.startswith("postgresql"):
            db_url = db_url.set(drivername="postgresql+asyncpg")
        else:
            logger.error(f"Unsupported database dialect in URL: {db_url.drivername}")
            return

        engine = create_async_engine(db_url)
        async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

        async with async_session() as session:
            logger.info("\n--- Database Statistics ---")
            await run_queries(session)

    except SQLAlchemyError as e:
        logger.error(f"A database error occurred: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
    finally:
        if engine:
            await engine.dispose()
            logger.info("Database connection closed.")


async def run_queries(session: AsyncSession):
    """Executes all the analysis queries within a single session."""

    # 1. Subject Cleanup (Consolidated Query)
    keyword_filters = ["%lunch%", "%booking%", "%out of office%", "%automatic reply%"]

    conditions = [Conversation.subject.ilike(kw) for kw in keyword_filters]

    # Create a series of CASE statements to count each keyword individually in one query
    cases = [func.sum(case((c, 1), else_=0)).label(kw.strip('%')) for c, kw in zip(conditions, keyword_filters)]

    stmt = select(*cases)

    results = (await session.execute(stmt)).first()

    logger.info("--- Conversation Subject Analysis ---")
    if results:
        for keyword, count in zip(keyword_filters, results):
            logger.info(f"Conversations matching '{keyword}': {count}")

    # 2. Audit Logs
    # NOTE: AuditLog.ts is a timezone-aware column (DateTime(timezone=True)).
    # Using a timezone-aware datetime object is correct.
    cutoff_30d = datetime.now(UTC) - THIRTY_DAYS
    stmt_old = select(func.count(AuditLog.audit_id)).where(AuditLog.ts < cutoff_30d)
    stmt_total = select(func.count(AuditLog.audit_id))

    old_audit_count = (await session.execute(stmt_old)).scalar_one()
    total_audit_count = (await session.execute(stmt_total)).scalar_one()

    logger.info("--- Audit Log Analysis ---")
    logger.info(f"Audit Logs: {total_audit_count} total, {old_audit_count} older than 30 days")

    # 3. Stuck Attachments
    # NOTE: Attachment.created_at is a timezone-aware column (DateTime(timezone=True)).
    # Using a timezone-aware datetime object is correct.
    cutoff_1h = datetime.now(UTC) - ONE_HOUR
    stmt_stuck = select(func.count(Attachment.attachment_id)).where(
        Attachment.status == "pending", Attachment.created_at < cutoff_1h
    )
    stuck_attachments = (await session.execute(stmt_stuck)).scalar_one()
    logger.info("--- Attachment Analysis ---")
    logger.info(f"Stuck Attachments (>1h pending): {stuck_attachments}")

    # 4. Orphaned Edges
    stmt_orphaned_edges = select(func.count(EntityEdge.edge_id)).where(
        EntityEdge.conversation_id.is_(None)
    )
    orphaned_edges = (await session.execute(stmt_orphaned_edges)).scalar_one()
    logger.info("--- Graph Analysis ---")
    logger.info(f"Orphaned Edges (No Conversation): {orphaned_edges}")

    # 5. Orphaned Nodes (ORM-based query)
    # Subquery to find all nodes that are part of at least one edge
    nodes_with_edges_subquery = select(EntityEdge.source_id).union(select(EntityEdge.target_id)).subquery()

    # Main query to find nodes whose ID is not in the subquery result
    stmt_isolated_nodes = select(func.count(EntityNode.node_id)).where(
        EntityNode.node_id.notin_(select(nodes_with_edges_subquery))
    )

    isolated_nodes = (await session.execute(stmt_isolated_nodes)).scalar_one()
    logger.info(f"Isolated Entities (No Edges): {isolated_nodes}")

    # 6. NULL Embeddings
    stmt_null_embeddings = select(func.count(Chunk.chunk_id)).where(Chunk.embedding.is_(None))
    null_embeddings = (await session.execute(stmt_null_embeddings)).scalar_one()
    logger.info("--- Chunk Analysis ---")
    logger.info(f"Chunks with NULL embeddings: {null_embeddings}")


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(analyze_db())
