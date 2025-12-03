"""
Mailroom: Orchestration entry for ingestion jobs.

Implements §6.1 of the Canonical Blueprint.
"""
from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from cortex.ingestion.conv_manifest.validation import Problem
from cortex.ingestion.parser_email import normalize_subject
from cortex.ingestion.pii import PIIInitError
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class IngestJob(BaseModel):
    """
    Ingestion job contract.

    Blueprint §6.1:
    * job_id: UUID
    * tenant_id: str
    * source_type: Literal["s3", "sftp", "local_upload"]
    * source_uri: str
    * options: Dict[str, Any]
    """

    job_id: uuid.UUID
    tenant_id: str
    source_type: Literal["s3", "sftp", "local_upload"]
    source_uri: str
    options: Dict[str, Any] = Field(default_factory=dict)


class IngestJobSummary(BaseModel):
    """
    Ingestion job summary.

    Blueprint §6.1:
    * job_id: UUID
    * tenant_id: str
    * messages_total: int
    * messages_ingested: int
    * messages_failed: int
    * attachments_total: int
    * attachments_parsed: int
    * attachments_failed: int
    * problems: List[Problem]
    * aborted_reason: Optional[str]
    """

    job_id: uuid.UUID
    tenant_id: str
    messages_total: int = 0
    messages_ingested: int = 0
    messages_failed: int = 0
    attachments_total: int = 0
    attachments_parsed: int = 0
    attachments_failed: int = 0
    problems: List[Problem] = Field(default_factory=list)
    aborted_reason: Optional[str] = None


def process_job(job: IngestJob) -> IngestJobSummary:
    """
    Process an ingestion job.

    This is the main entry point for the ingestion pipeline.
    It orchestrates:
    1. Fetching data from source
    2. Parsing emails/threads
    3. PII redaction
    4. Attachment extraction
    5. Writing to DB
    """
    summary = IngestJobSummary(job_id=job.job_id, tenant_id=job.tenant_id)

    logger.info(f"Starting ingestion job {job.job_id} for tenant {job.tenant_id}")

    try:
        # 1. Validate source (basic check)
        if not job.source_uri:
            raise ValueError("Source URI is required")

        # 2. Download/Stream content (Placeholder for S3/SFTP)

        if job.source_type == "local_upload":
            import uuid
            from datetime import datetime, timezone
            from pathlib import Path

            from cortex.chunking.chunker import ChunkingInput, Span, chunk_text
            from cortex.config.loader import get_config
            from cortex.db.session import SessionLocal
            from cortex.embeddings.client import EmbeddingsClient
            from cortex.ingestion.conv_loader import load_conversation
            from cortex.ingestion.quoted_masks import detect_quoted_spans
            from cortex.ingestion.text_preprocessor import \
                get_text_preprocessor
            from cortex.ingestion.writer import DBWriter

            config = get_config()
            preprocessor = get_text_preprocessor()

            convo_dir = Path(job.source_uri)
            if not convo_dir.exists():
                raise ValueError(f"Local path does not exist: {job.source_uri}")

            # Load conversation
            convo_data = load_conversation(convo_dir, include_attachment_text=True)
            if not convo_data:
                logger.warning(f"No data loaded from {convo_dir}")
                return summary

            logger.info(f"Loaded conversation: {convo_data.get('path')}")

            # --- Transformation Logic ---

            # 1. Thread
            thread_id = uuid.uuid4()
            now = datetime.now(timezone.utc)

            # Try to get subject/dates from manifest or summary
            manifest = convo_data.get("manifest", {})
            summary_json = convo_data.get("summary", {})

            subject = (
                manifest.get("subject_label")
                or summary_json.get("subject")
                or convo_dir.name
            )
            subject_norm = normalize_subject(subject)

            # Parse dates from manifest if available
            created_at = now
            updated_at = now

            if manifest.get("started_at_utc"):
                try:
                    from dateutil import parser as date_parser

                    created_at = date_parser.parse(manifest["started_at_utc"])
                except Exception:
                    pass

            if manifest.get("ended_at_utc"):
                try:
                    from dateutil import parser as date_parser

                    updated_at = date_parser.parse(manifest["ended_at_utc"])
                except Exception:
                    pass

            thread_data = {
                "thread_id": thread_id,
                "subject_norm": subject_norm,
                "original_subject": subject,
                "created_at": created_at,
                "updated_at": updated_at,
                "metadata": {"source_path": str(convo_dir)},
            }

            # 2. Message (Single transcript for now)
            message_id = str(uuid.uuid4())

            # Preprocess Body (§6.8)
            raw_body = convo_data.get("conversation_txt", "")
            cleaned_body, body_meta = preprocessor.prepare_for_indexing(
                raw_body, text_type="email", tenant_id=job.tenant_id
            )

            # Detect Quoted Spans (§6.3)
            quoted_spans_dicts = detect_quoted_spans(cleaned_body)
            quoted_spans = [Span(**s) for s in quoted_spans_dicts]
            has_quoted_mask = len(quoted_spans) > 0

            message_data = {
                "message_id": message_id,
                "thread_id": thread_id,
                "from_addr": "system@ingest",  # Placeholder
                "to_addrs": [],
                "subject": subject,
                "body_plain": cleaned_body,
                "sent_at": now,
                "has_quoted_mask": has_quoted_mask,
                "metadata": {
                    "type": "transcript",
                    "quoted_spans": quoted_spans_dicts,
                    **body_meta,
                },
            }

            # 3. Attachments
            attachments_data = []
            for att in convo_data.get("attachments", []):
                att_id = uuid.uuid4()

                # Preprocess Attachment Text (§6.8)
                raw_att_text = att.get("text", "")
                cleaned_att_text, att_meta = preprocessor.prepare_for_indexing(
                    raw_att_text, text_type="attachment", tenant_id=job.tenant_id
                )

                attachments_data.append(
                    {
                        "attachment_id": att_id,
                        "message_id": message_id,
                        "filename": Path(att["path"]).name,
                        "storage_uri_raw": att["path"],
                        "status": "parsed",
                        "extracted_chars": len(cleaned_att_text),
                        "text": cleaned_att_text,  # Keep text for chunking
                        "metadata": att_meta,
                    }
                )

            # 4. Chunking & Embedding
            chunks_data = []
            texts_to_embed = []
            chunk_refs = []  # (chunk_dict, index_in_texts)

            # Chunk message body
            if message_data["body_plain"]:
                body_chunks = chunk_text(
                    ChunkingInput(
                        text=message_data["body_plain"],
                        section_path="email:body",
                        quoted_spans=quoted_spans,
                    )
                )
                for c in body_chunks:
                    c_dict = c.model_dump()
                    c_dict["chunk_id"] = uuid.uuid4()
                    c_dict["thread_id"] = thread_id
                    c_dict["message_id"] = message_id
                    c_dict["tenant_id"] = job.tenant_id

                    chunks_data.append(c_dict)
                    texts_to_embed.append(c.text)
                    chunk_refs.append(c_dict)

            # Chunk attachments
            for att in attachments_data:
                if att["text"]:
                    att_chunks = chunk_text(
                        ChunkingInput(
                            text=att["text"],
                            section_path=f"attachment:{att['filename']}",
                        )
                    )
                    for c in att_chunks:
                        c_dict = c.model_dump()
                        c_dict["chunk_id"] = uuid.uuid4()
                        c_dict["thread_id"] = thread_id
                        c_dict["message_id"] = message_id
                        c_dict["attachment_id"] = att["attachment_id"]
                        c_dict["tenant_id"] = job.tenant_id

                        chunks_data.append(c_dict)
                        texts_to_embed.append(c.text)
                        chunk_refs.append(c_dict)

            # Embed
            if texts_to_embed:
                logger.info(f"Embedding {len(texts_to_embed)} chunks...")
                client = EmbeddingsClient()
                embeddings = client.embed_batch(texts_to_embed)

                for i, emb in enumerate(embeddings):
                    chunk_refs[i]["embedding"] = emb
                    chunk_refs[i]["embedding_model"] = config.embedding.model_name

            # 5. Write
            transformed_results = {
                "threads": [thread_data],
                "messages": [message_data],
                "attachments": attachments_data,
                "chunks": chunks_data,
            }

            with SessionLocal() as session:
                writer = DBWriter(session)
                writer.write_job_results(job, transformed_results)

            summary.messages_total = 1
            summary.messages_ingested = 1
            summary.attachments_total = len(attachments_data)
            summary.attachments_parsed = len(attachments_data)

        else:
            logger.warning(f"Unsupported source type: {job.source_type}")

    except PIIInitError as e:
        logger.critical(f"Ingestion job aborted: PII engine failed to initialize. {e}")
        summary.aborted_reason = "pii_init_failed"
        summary.problems.append(Problem(folder=job.source_uri, issue="pii_init_failed"))

    except Exception as e:
        logger.error(f"Ingestion job failed: {e}", exc_info=True)
        summary.aborted_reason = str(e)
        summary.problems.append(Problem(folder=job.source_uri, issue="job_failed"))

    return summary
