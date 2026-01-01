"""
Rechunk skipped chunks that failed earlier ingestion steps.
"""

from __future__ import annotations

import logging
from typing import Any

from cortex.chunking.chunker import ChunkingInput, chunk_text
from cortex.config.loader import get_config
from cortex.db.models import Chunk
from cortex.db.session import SessionLocal
from sqlalchemy import select

logger = logging.getLogger(__name__)


def _build_extra_data(bad_chunk: Chunk) -> dict[str, Any]:
    extra_data = dict(bad_chunk.extra_data) if bad_chunk.extra_data else {}
    extra_data.pop("skipped", None)
    extra_data.pop("error", None)
    extra_data["rechunked_from"] = str(bad_chunk.chunk_id)
    return extra_data


def rechunk_failed(
    *,
    tenant_id: str | None = None,
    max_tokens: int | None = None,
    overlap_tokens: int | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    """
    Rechunk chunks flagged as skipped in extra_data.

    Returns a summary with counts and optional error details.
    """
    results: dict[str, Any] = {
        "rechunked": 0,
        "deleted": 0,
        "skipped": 0,
        "error": None,
    }
    session = SessionLocal()
    try:
        config = get_config().processing
        safe_max_tokens = max_tokens if max_tokens is not None else config.chunk_size
        safe_overlap = (
            overlap_tokens if overlap_tokens is not None else config.chunk_overlap
        )

        stmt = select(Chunk).where(Chunk.extra_data.op("->>")("skipped") == "true")
        if tenant_id:
            stmt = stmt.where(Chunk.tenant_id == tenant_id)
        if isinstance(limit, int) and limit > 0:
            stmt = stmt.limit(limit)

        bad_chunks = session.execute(stmt).scalars().all()
        if not bad_chunks:
            return results

        for bad_chunk in bad_chunks:
            if not bad_chunk.text:
                results["skipped"] += 1
                continue

            input_data = ChunkingInput(
                text=bad_chunk.text,
                section_path=bad_chunk.section_path or "unknown",
                max_tokens=safe_max_tokens,
                overlap_tokens=safe_overlap,
                quoted_spans=[],
            )
            new_models = chunk_text(input_data)
            if not new_models:
                results["skipped"] += 1
                continue

            base_char_start = (
                bad_chunk.char_start if isinstance(bad_chunk.char_start, int) else 0
            )
            base_position = (
                bad_chunk.position if isinstance(bad_chunk.position, int) else 0
            )
            extra_data = _build_extra_data(bad_chunk)

            for model in new_models:
                new_chunk = Chunk(
                    tenant_id=bad_chunk.tenant_id,
                    conversation_id=bad_chunk.conversation_id,
                    attachment_id=bad_chunk.attachment_id,
                    is_attachment=bad_chunk.is_attachment,
                    is_summary=bad_chunk.is_summary,
                    chunk_type=model.chunk_type or bad_chunk.chunk_type,
                    text=model.text,
                    position=base_position + model.position,
                    char_start=base_char_start + model.char_start,
                    char_end=base_char_start + model.char_end,
                    section_path=bad_chunk.section_path,
                    extra_data=extra_data,
                )
                session.add(new_chunk)
                results["rechunked"] += 1

            session.delete(bad_chunk)
            results["deleted"] += 1

        if results["rechunked"] or results["deleted"]:
            session.commit()

        return results
    except Exception as exc:
        logger.error("Failed to rechunk skipped chunks: %s", exc, exc_info=True)
        session.rollback()
        results["error"] = str(exc)
        return results
    finally:
        session.close()
