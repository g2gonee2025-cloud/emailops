"""
Mailroom: Orchestration entry for ingestion jobs.

Implements §6.1 of the Canonical Blueprint.
Updated for Conversation-based schema.
"""
from __future__ import annotations

import logging
import os
import shutil
import stat
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from cortex.ingestion.core_manifest import resolve_subject
from cortex.ingestion.models import IngestJobRequest, IngestJobSummary, Problem
from cortex.ingestion.pii import PIIInitError

logger = logging.getLogger(__name__)


def process_job(job: IngestJobRequest) -> IngestJobSummary:
    """
    Process an ingestion job.

    This is the main entry point for the ingestion pipeline.
    It orchestrates:
    1. Fetching data from source
    2. Parsing emails/threads
    3. PII redaction
    4. Attachment extraction
    5. Chunking (no embedding by request)
    6. Writing to DB
    """
    summary = IngestJobSummary(job_id=job.job_id, tenant_id=job.tenant_id)

    logger.info(f"Starting ingestion job {job.job_id} for tenant {job.tenant_id}")

    try:
        if not job.source_uri:
            raise ValueError("Source URI is required")

        local_convo_dir: Optional[Path] = None
        temp_dir: Optional[Path] = None

        try:
            if job.source_type == "local_upload":
                local_convo_dir = _validate_local_path(job.source_uri)
            elif job.source_type == "s3":
                temp_dir, local_convo_dir = _download_s3_source(job.source_uri)
            elif job.source_type == "sftp":
                temp_dir, local_convo_dir = _download_sftp_source(
                    job.source_uri, job.options
                )
            else:
                raise ValueError(f"Unsupported source type: {job.source_type}")

            summary = _ingest_conversation(local_convo_dir, job, summary)

        finally:
            if temp_dir and temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)

    except PIIInitError as e:
        logger.critical(f"Ingestion job aborted: PII engine failed to initialize. {e}")
        summary.aborted_reason = "pii_init_failed"
        summary.problems.append(Problem(folder=job.source_uri, issue="pii_init_failed"))

    except Exception as e:
        logger.error(f"Ingestion job failed: {e}", exc_info=True)
        summary.aborted_reason = str(e)
        summary.problems.append(Problem(folder=job.source_uri, issue="job_failed"))

    return summary


def _validate_local_path(path_str: str) -> Path:
    path = Path(path_str)
    if not path.exists():
        raise ValueError(f"Local path does not exist: {path_str}")
    if not path.is_dir():
        raise ValueError(f"Path is not a directory: {path_str}")
    return path


def _normalize_s3_prefix(source_uri: str, bucket: str) -> Tuple[str, Optional[str]]:
    """Return prefix and optional bucket extracted from a URI or raw prefix."""
    parsed = urlparse(source_uri)
    if parsed.scheme in {"s3", "spaces"}:
        prefix = parsed.path.lstrip("/")
        return (prefix if prefix else "", parsed.netloc or bucket)
    return source_uri.lstrip("/"), None


def _download_s3_source(source_uri: str) -> Tuple[Path, Path]:
    from cortex.ingestion.s3_source import create_s3_source

    handler = create_s3_source()
    prefix, bucket_from_uri = _normalize_s3_prefix(source_uri, handler.bucket)

    if bucket_from_uri and bucket_from_uri != handler.bucket:
        logger.warning(
            "Source bucket %s does not match configured bucket %s; using configured bucket",
            bucket_from_uri,
            handler.bucket,
        )

    folder = handler.build_folder(prefix)
    if not folder.files:
        raise ValueError(f"No files found at prefix: {prefix}")

    temp_dir = Path(tempfile.mkdtemp(prefix="cortex_s3_"))
    local_dir = handler.download_conversation_folder(folder, temp_dir)
    return temp_dir, local_dir


def _download_sftp_source(
    source_uri: str, options: Dict[str, Any]
) -> Tuple[Path, Path]:
    parsed = urlparse(source_uri if "://" in source_uri else f"sftp://{source_uri}")
    if parsed.scheme and parsed.scheme != "sftp":
        raise ValueError(f"Invalid SFTP URI: {source_uri}")

    host = parsed.hostname or options.get("host")
    remote_path = parsed.path or options.get("path")
    username = parsed.username or options.get("username")
    password = parsed.password or options.get("password")
    port = parsed.port or int(options.get("port", 22))
    pkey_path = options.get("pkey_path")

    if not host or not remote_path:
        raise ValueError("SFTP source requires host and path")

    try:
        import paramiko  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ValueError(
            "paramiko is required for sftp ingestion. Install it to use source_type='sftp'."
        ) from exc

    from cortex.config.loader import get_config

    config = get_config()

    client = paramiko.SSHClient()

    if config.core.env == "prod":
        client.load_system_host_keys()
        client.set_missing_host_key_policy(paramiko.RejectPolicy())
    else:
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    pkey = None
    if pkey_path:
        pkey = paramiko.RSAKey.from_private_key_file(pkey_path)

    sftp = None
    try:
        client.connect(
            hostname=host,
            port=port,
            username=username,
            password=password,
            pkey=pkey,
            timeout=30,
        )

        sftp = client.open_sftp()

        temp_dir = Path(tempfile.mkdtemp(prefix="cortex_sftp_"))
        remote_root = Path(remote_path)
        root_name = remote_root.name or remote_root.parent.name or "sftp_conversation"
        local_root = temp_dir / root_name
        local_root.mkdir(parents=True, exist_ok=True)

        _download_sftp_tree(sftp, str(remote_root), local_root)

        return temp_dir, local_root
    finally:
        if sftp is not None:
            try:
                sftp.close()
            except Exception:
                pass
        client.close()


def _download_sftp_tree(sftp: Any, remote_path: str, local_path: Path) -> None:
    """Recursively download an SFTP directory into local_path."""
    for entry in sftp.listdir_attr(remote_path):
        remote_child = os.path.join(remote_path, entry.filename)
        local_child = local_path / entry.filename

        if stat.S_ISDIR(entry.st_mode):
            local_child.mkdir(parents=True, exist_ok=True)
            _download_sftp_tree(sftp, remote_child, local_child)
        else:
            local_child.parent.mkdir(parents=True, exist_ok=True)
            sftp.get(remote_child, str(local_child))


def _ingest_conversation(
    convo_dir: Path, job: IngestJobRequest, summary: IngestJobSummary
) -> IngestJobSummary:
    """
    Ingest a conversation folder into the database.

    Uses new Conversation-based schema (no Thread/Message split).
    """
    from cortex.chunking.chunker import ChunkingInput, Span, chunk_text
    from cortex.config.loader import get_config
    from cortex.db.session import SessionLocal, set_session_tenant
    from cortex.ingestion.attachments_log import parse_attachments_log
    from cortex.ingestion.conv_loader import load_conversation
    from cortex.ingestion.quoted_masks import detect_quoted_spans
    from cortex.ingestion.text_preprocessor import get_text_preprocessor
    from cortex.ingestion.writer import DBWriter

    config = get_config()
    preprocessor = get_text_preprocessor()

    # Load conversation data from disk
    convo_data = load_conversation(convo_dir, include_attachment_text=True)
    if not convo_data:
        logger.warning("No data loaded from %s", convo_dir)
        return summary

    logger.info("Loaded conversation: %s", convo_data.get("path"))

    # Load attachment metadata from log CSV if available
    att_log_meta = parse_attachments_log(convo_dir)

    # Generate IDs
    conversation_id = uuid.uuid4()
    now = datetime.now(timezone.utc)

    # Extract manifest and summary data
    manifest = convo_data.get("manifest", {})
    summary_json = convo_data.get("summary", {})

    subject, subject_norm = resolve_subject(manifest, summary_json, convo_dir.name)

    # Parse dates
    earliest_date = now
    latest_date = now

    if manifest.get("started_at_utc"):
        try:
            from dateutil import parser as date_parser

            earliest_date = date_parser.parse(manifest["started_at_utc"])
        except Exception:
            pass

    if manifest.get("ended_at_utc"):
        try:
            from dateutil import parser as date_parser

            latest_date = date_parser.parse(manifest["ended_at_utc"])
        except Exception:
            pass

    # Extract participants from Conversation.txt (manifest doesn't have messages array)
    from cortex.ingestion.conversation_parser import (
        extract_participants_from_conversation_txt,
    )

    raw_body = convo_data.get("conversation_txt", "")
    participants = extract_participants_from_conversation_txt(raw_body)

    # Build conversation data
    conversation_data: Dict[str, Any] = {
        "conversation_id": conversation_id,
        "folder_name": convo_dir.name,
        "subject": subject_norm or subject,
        "smart_subject": manifest.get("smart_subject"),
        "participants": participants,
        "messages": [],  # Manifest doesn't contain structured messages
        "earliest_date": earliest_date,
        "latest_date": latest_date,
        "storage_uri": str(convo_dir),
        "extra_data": {
            "source_path": str(convo_dir),
            "ingest_source": job.source_type,
        },
    }

    # Process body text
    raw_body = convo_data.get("conversation_txt", "")
    cleaned_body, body_meta = preprocessor.prepare_for_indexing(
        raw_body, text_type="email", tenant_id=job.tenant_id
    )

    # Detect quoted spans for chunk classification
    quoted_spans: List[Span] = detect_quoted_spans(cleaned_body)

    # Process attachments
    attachments_data: List[Dict[str, Any]] = []
    for att in convo_data.get("attachments", []):
        att_id = uuid.uuid4()
        filename = Path(att["path"]).name

        # Skip metadata/generated files that shouldn't be chunked
        skip_filenames = {"attachments_log.csv", "Conversation_human.txt"}
        if filename in skip_filenames:
            continue

        raw_att_text = att.get("text", "")
        cleaned_att_text, att_meta = preprocessor.prepare_for_indexing(
            raw_att_text, text_type="attachment", tenant_id=job.tenant_id
        )

        # Enrich with log metadata if available
        if filename in att_log_meta:
            att_meta.update(att_log_meta[filename])

        attachments_data.append(
            {
                "attachment_id": att_id,
                "conversation_id": conversation_id,
                "filename": filename,
                "content_type": att_meta.get("mime_type"),
                "size_bytes": att_meta.get("size_bytes"),
                "storage_uri": att.get("storage_uri_raw", att["path"]),
                "status": "parsed",
                "text": cleaned_att_text,  # For chunking, not stored in DB
            }
        )

    # Chunk body text
    chunks_data: List[Dict[str, Any]] = []

    if cleaned_body:
        body_chunks = chunk_text(
            ChunkingInput(
                text=cleaned_body,
                section_path="email:body",
                quoted_spans=quoted_spans,
                max_tokens=config.processing.chunk_size,
                overlap_tokens=config.processing.chunk_overlap,
            )
        )
        for c in body_chunks:
            chunks_data.append(
                {
                    "chunk_id": uuid.uuid4(),
                    "conversation_id": conversation_id,
                    "is_attachment": False,
                    "text": c.text,
                    "position": c.position,
                    "char_start": c.char_start,
                    "char_end": c.char_end,
                    "section_path": c.section_path,
                    "extra_data": c.metadata,
                }
            )

    # Chunk attachments
    for att in attachments_data:
        if att.get("text"):
            att_chunks = chunk_text(
                ChunkingInput(
                    text=att["text"],
                    section_path=f"attachment:{att['filename']}",
                    max_tokens=config.processing.chunk_size,
                    overlap_tokens=config.processing.chunk_overlap,
                )
            )
            for c in att_chunks:
                chunks_data.append(
                    {
                        "chunk_id": uuid.uuid4(),
                        "conversation_id": conversation_id,
                        "attachment_id": att["attachment_id"],
                        "is_attachment": True,
                        "text": c.text,
                        "position": c.position,
                        "char_start": c.char_start,
                        "char_end": c.char_end,
                        "section_path": c.section_path,
                        "extra_data": c.metadata,
                    }
                )

    # Remove text from attachments (not stored in DB, only used for chunking)
    for att in attachments_data:
        att.pop("text", None)

    # Build results
    results: Dict[str, Any] = {
        "conversation": conversation_data,
        "attachments": attachments_data,
        "chunks": chunks_data,
    }

    # Write to database
    logger.info(
        f"Writing: 1 conversation, {len(attachments_data)} attachments, {len(chunks_data)} chunks"
    )
    with SessionLocal() as session:
        set_session_tenant(session, job.tenant_id)
        writer = DBWriter(session)
        writer.write_job_results(job, results)

    # Update summary
    summary.messages_total = manifest.get("message_count", 1)
    summary.messages_ingested = manifest.get("message_count", 1)
    summary.attachments_total = len(attachments_data)
    summary.attachments_parsed = len(attachments_data)
    summary.chunks_created = len(chunks_data)
    summary.threads_created = 1

    return summary
