
import logging
import os
from sqlalchemy import select
from cortex.db.session import SessionLocal
from cortex.db.models import Chunk
from cortex.chunking.chunker import chunk_text, ChunkingInput

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rechunk")

def rechunk_failed():
    session = SessionLocal()
    # Find skipped chunks
    try:
        stmt = select(Chunk).where(Chunk.extra_data.op("->>")("skipped") == "true")
        bad_chunks = session.execute(stmt).scalars().all()

        logger.info(f"Found {len(bad_chunks)} skipped chunks to re-process.")

        if not bad_chunks:
            logger.info("No skipped chunks found. Exiting.")
            return

        total_new = 0

        for bad_chunk in bad_chunks:
            logger.info(f"Processing chunk {bad_chunk.chunk_id} (Length: {len(bad_chunk.text)} chars)")

            # Use the NEW chunker logic which respects token limits
            inp = ChunkingInput(
                text=bad_chunk.text,
                section_path=bad_chunk.section_path or "unknown",
                max_tokens=1600,
                overlap_tokens=200,
                quoted_spans=[]
            )

            new_models = chunk_text(inp)
            logger.info(f"  -> Splitting into {len(new_models)} new valid chunks.")

            for model in new_models:
                new_chunk = Chunk(
                    conversation_id=bad_chunk.conversation_id,
                    attachment_id=bad_chunk.attachment_id,
                    is_attachment=bad_chunk.is_attachment,
                    text=model.text,
                    # Adjust position relative to original, though purely appending is fine for retrieval
                    position=bad_chunk.position,
                    char_start=bad_chunk.char_start + model.char_start,
                    char_end=bad_chunk.char_start + model.char_end,
                    section_path=bad_chunk.section_path,
                    extra_data={"rechunked_from": str(bad_chunk.chunk_id)}
                )
                session.add(new_chunk)
                total_new += 1

            # Remove the bad chunk
            session.delete(bad_chunk)

        session.commit()
        logger.info(f"Success! Replaced {len(bad_chunks)} bad chunks with {total_new} valid chunks.")

    except Exception as e:
        logger.error(f"Failed to rechunk: {e}")
        session.rollback()
    finally:
        session.close()

if __name__ == "__main__":
    rechunk_failed()
