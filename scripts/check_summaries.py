import asyncio
import sys
from pathlib import Path

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

# Ensure backend path is in sys.path
sys.path.append(str(Path("backend/src").resolve()))

from cortex.config.loader import get_config
from cortex.db.models import Conversation


async def check_summaries():
    config = get_config()
    db_url = config.database.url.replace("postgresql://", "postgresql+asyncpg://")
    if "sslmode=require" in db_url:
        db_url = db_url.replace("sslmode=require", "ssl=require")

    print("Connecting to database...")
    engine = create_async_engine(db_url)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with async_session() as session:
        print("\n--- Summary Verification ---")

        # 1. Total Conversations
        stmt_total = select(func.count(Conversation.conversation_id))
        total_count = (await session.execute(stmt_total)).scalar()

        # 2. Missing Summaries (NULL)
        stmt_null = select(func.count(Conversation.conversation_id)).where(
            Conversation.summary_text.is_(None)
        )
        null_count = (await session.execute(stmt_null)).scalar()

        # 3. Empty Summaries ("")
        stmt_empty = select(func.count(Conversation.conversation_id)).where(
            Conversation.summary_text == ""
        )
        empty_count = (await session.execute(stmt_empty)).scalar()

        # 4. Valid Summaries
        valid_count = total_count - null_count - empty_count

        print(f"Total Conversations: {total_count}")
        print(f"Valid Summaries:     {valid_count}")
        print(f"Missing (NULL):      {null_count}")
        print(f"Empty Strings:       {empty_count}")

        if total_count > 0:
            coverage = (valid_count / total_count) * 100
            print(f"Coverage:            {coverage:.2f}%")
        else:
            print("Coverage: N/A (No conversations found)")

        if null_count > 0 or empty_count > 0:
            print("\nWARNING: Some conversations are missing summaries.")

            # optional: list a few IDs that are missing
            stmt_missing_ids = (
                select(Conversation.conversation_id)
                .where(
                    (Conversation.summary_text.is_(None))
                    | (Conversation.summary_text == "")
                )
                .limit(5)
            )
            missing_ids = (await session.execute(stmt_missing_ids)).scalars().all()
            print(f"Sample IDs missing summaries: {missing_ids}")

    await engine.dispose()


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(check_summaries())
