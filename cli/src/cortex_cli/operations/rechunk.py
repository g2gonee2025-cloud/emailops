"""
Re-chunking operation for Cortex CLI.
Finds oversized chunks and re-chunks them into smaller, valid chunks.
"""

import logging
from collections.abc import Callable
from typing import Any

from cortex.chunking.chunker import ChunkingInput, chunk_text
from cortex.db.models import Chunk
from cortex.db.session import SessionLocal
from sqlalchemy import func, select

logger = logging.getLogger(__name__)


def run_rechunk(
    tenant_id: str | None,
    chunk_size_limit: int,
    dry_run: bool,
    max_tokens: int,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """
    Finds and re-chunks oversized chunks.
    """
    session = SessionLocal()
    results = {
        "oversized_chunks_found": 0,
        "new_chunks_created": 0,
        "chunks_deleted": 0,
        "dry_run": dry_run,
    }

    try:
        # Find oversized chunks
        stmt = select(Chunk).where(func.length(Chunk.text) > chunk_size_limit)
        if tenant_id:
            stmt = stmt.where(Chunk.tenant_id == tenant_id)

        bad_chunks = session.execute(stmt).scalars().all()
        results["oversized_chunks_found"] = len(bad_chunks)

        if not bad_chunks:
            return results

        total_new = 0
        for bad_chunk in bad_chunks:
            inp = ChunkingInput(
                text=bad_chunk.text,
                section_path=bad_chunk.section_path or "unknown",
                max_tokens=max_tokens,
                overlap_tokens=200,
                quoted_spans=[],
            )

            new_models = chunk_text(inp)

            for model in new_models:
                new_chunk = Chunk(
                    tenant_id=bad_chunk.tenant_id,
                    conversation_id=bad_chunk.conversation_id,
                    attachment_id=bad_chunk.attachment_id,
                    is_attachment=bad_chunk.is_attachment,
                    text=model.text,
                    position=bad_chunk.position,
                    char_start=bad_chunk.char_start + model.char_start,
                    char_end=bad_chunk.char_start + model.char_end,
                    section_path=bad_chunk.section_path,
                    extra_data={"rechunked_from": str(bad_chunk.chunk_id)},
                )
                if not dry_run:
                    session.add(new_chunk)
                total_new += 1

            if not dry_run:
                session.delete(bad_chunk)
            results["chunks_deleted"] += 1

            if progress_callback:
                progress_callback(
                    {
                        "processed": results["chunks_deleted"],
                        "total": len(bad_chunks),
                        "new": total_new,
                    }
                )

        results["new_chunks_created"] = total_new

        if not dry_run:
            session.commit()

    except Exception as e:
        logger.error(f"Failed to rechunk: {e}")
        if not dry_run:
            session.rollback()
        raise e
    finally:
        session.close()

    return results
