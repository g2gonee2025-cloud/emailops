"""
Re-chunking operation for Cortex CLI.
Finds oversized chunks and re-chunks them into smaller, valid chunks.
"""

import logging
from collections.abc import Callable
from typing import Any

from cortex.chunking.chunker import ChunkingInput, TokenCounter, chunk_text
from cortex.db.models import Chunk
from cortex.db.session import SessionLocal
from sqlalchemy import func, select

logger = logging.getLogger(__name__)

DEFAULT_OVERLAP_TOKENS = 200
BATCH_SIZE = 500


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
        normalized_tenant = None
        if tenant_id is not None:
            if not isinstance(tenant_id, str):
                raise ValueError("tenant_id must be a string")
            normalized_tenant = tenant_id.strip()
            if not normalized_tenant:
                raise ValueError("tenant_id must be non-empty when provided")

        token_counter = TokenCounter()
        approx_token_chars = token_counter.tokens_to_chars(max_tokens)
        prefilter_chars = max(int(chunk_size_limit), int(approx_token_chars))

        # Use stored char offsets to avoid length() scans on text.
        char_span = Chunk.char_end - Chunk.char_start
        stmt = select(Chunk).where(char_span >= prefilter_chars)
        count_stmt = (
            select(func.count()).select_from(Chunk).where(char_span >= prefilter_chars)
        )
        if normalized_tenant is not None:
            stmt = stmt.where(Chunk.tenant_id == normalized_tenant)
            count_stmt = count_stmt.where(Chunk.tenant_id == normalized_tenant)

        candidate_total = session.execute(count_stmt).scalar_one()

        total_new = 0
        total_deleted = 0
        processed = 0
        stream = session.execute(stmt.execution_options(yield_per=BATCH_SIZE)).scalars()
        for bad_chunk in stream:
            processed += 1
            token_count = token_counter.count(bad_chunk.text or "")
            if token_count <= max_tokens:
                if progress_callback:
                    try:
                        progress_callback(
                            {
                                "processed": processed,
                                "total": candidate_total,
                                "new": total_new,
                            }
                        )
                    except Exception as callback_exc:
                        logger.warning(
                            "Progress callback failed: %s",
                            callback_exc,
                            exc_info=True,
                        )
                continue

            results["oversized_chunks_found"] += 1
            inp = ChunkingInput(
                text=bad_chunk.text,
                section_path=bad_chunk.section_path or "unknown",
                max_tokens=max_tokens,
                overlap_tokens=DEFAULT_OVERLAP_TOKENS,
                quoted_spans=[],
            )

            new_models = chunk_text(inp)
            if not new_models:
                logger.warning(
                    "Skipping rechunk for %s; no new chunks produced",
                    bad_chunk.chunk_id,
                )
                continue

            base_char_start = (
                bad_chunk.char_start if isinstance(bad_chunk.char_start, int) else 0
            )
            base_position = (
                bad_chunk.position if isinstance(bad_chunk.position, int) else 0
            )

            for model in new_models:
                new_chunk = Chunk(
                    tenant_id=bad_chunk.tenant_id,
                    conversation_id=bad_chunk.conversation_id,
                    attachment_id=bad_chunk.attachment_id,
                    is_attachment=bad_chunk.is_attachment,
                    text=model.text,
                    position=base_position + model.position,
                    char_start=base_char_start + model.char_start,
                    char_end=base_char_start + model.char_end,
                    section_path=bad_chunk.section_path,
                    extra_data={"rechunked_from": str(bad_chunk.chunk_id)},
                )
                if not dry_run:
                    session.add(new_chunk)
                total_new += 1

            if not dry_run:
                session.delete(bad_chunk)
                total_deleted += 1

            if progress_callback:
                try:
                    progress_callback(
                        {
                            "processed": processed,
                            "total": candidate_total,
                            "new": total_new,
                        }
                    )
                except Exception as callback_exc:
                    logger.warning(
                        "Progress callback failed: %s",
                        callback_exc,
                        exc_info=True,
                    )

        if not dry_run:
            session.commit()
            results["new_chunks_created"] = total_new
            results["chunks_deleted"] = total_deleted

    except Exception:
        logger.error("Failed to rechunk", exc_info=True)
        if not dry_run:
            session.rollback()
        raise
    finally:
        session.close()

    return results
