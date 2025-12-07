"""
Mailroom: Orchestration entry for ingestion jobs.

Implements §6.1 of the Canonical Blueprint.
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
from typing import Any, Dict, List, Literal, Optional, Tuple
from urllib.parse import urlparse

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
    problems: List[Problem] = Field(default_factory=lambda: [])
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

        # 2. Download/stream content based on source type
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

    client = paramiko.SSHClient()
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


def _resolve_sender(config: Any) -> str:
    candidates = [
        config.email.sender_locked_email,
        config.email.sender_reply_to,
    ]
    if config.email.message_id_domain:
        candidates.append(f"no-reply@{config.email.message_id_domain}")
    candidates.append("no-reply@emailops.local")
    for candidate in candidates:
        if candidate:
            return candidate
    return "no-reply@emailops.local"


def _ingest_conversation(
    convo_dir: Path, job: IngestJob, summary: IngestJobSummary
) -> IngestJobSummary:
    from cortex.chunking.chunker import ChunkingInput, Span, chunk_text
    from cortex.config.loader import get_config
    from cortex.db.session import SessionLocal, set_session_tenant
    from cortex.embeddings.client import EmbeddingsClient
    from cortex.ingestion.conv_loader import load_conversation
    from cortex.ingestion.quoted_masks import detect_quoted_spans
    from cortex.ingestion.text_preprocessor import get_text_preprocessor
    from cortex.ingestion.writer import DBWriter

    config = get_config()
    preprocessor = get_text_preprocessor()

    convo_data = load_conversation(convo_dir, include_attachment_text=True)
    if not convo_data:
        logger.warning("No data loaded from %s", convo_dir)
        return summary

    logger.info("Loaded conversation: %s", convo_data.get("path"))

    thread_id = uuid.uuid4()
    now = datetime.now(timezone.utc)

    manifest = convo_data.get("manifest", {})
    summary_json = convo_data.get("summary", {})

    subject = (
        manifest.get("subject_label") or summary_json.get("subject") or convo_dir.name
    )
    subject_norm = normalize_subject(subject)

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

    thread_data: Dict[str, Any] = {
        "thread_id": thread_id,
        "subject_norm": subject_norm,
        "original_subject": subject,
        "created_at": created_at,
        "updated_at": updated_at,
        "metadata": {"source_path": str(convo_dir)},
    }

    message_id = str(uuid.uuid4())
    raw_body = convo_data.get("conversation_txt", "")
    cleaned_body, body_meta = preprocessor.prepare_for_indexing(
        raw_body, text_type="email", tenant_id=job.tenant_id
    )

    quoted_spans_dicts = detect_quoted_spans(cleaned_body)
    quoted_spans = [Span(**s) for s in quoted_spans_dicts]
    has_quoted_mask = len(quoted_spans) > 0

    from_addr = _resolve_sender(config)

    message_data: Dict[str, Any] = {
        "message_id": message_id,
        "thread_id": thread_id,
        "from_addr": from_addr,
        "to_addrs": [],
        "subject": subject,
        "body_plain": cleaned_body,
        "sent_at": now,
        "has_quoted_mask": has_quoted_mask,
        "metadata": {
            "type": "transcript",
            "quoted_spans": quoted_spans_dicts,
            "ingest_source": job.source_type,
            **body_meta,
        },
    }

    attachments_data: List[Dict[str, Any]] = []
    for att in convo_data.get("attachments", []):
        att_id = uuid.uuid4()

        raw_att_text = att.get("text", "")
        cleaned_att_text, att_meta = preprocessor.prepare_for_indexing(
            raw_att_text, text_type="attachment", tenant_id=job.tenant_id
        )

        attachments_data.append(
            {
                "attachment_id": att_id,
                "message_id": message_id,
                "filename": Path(att["path"]).name,
                "storage_uri_raw": att.get("storage_uri_raw", att["path"]),
                "status": "parsed",
                "extracted_chars": len(cleaned_att_text),
                "text": cleaned_att_text,
                "metadata": att_meta,
            }
        )

    chunks_data: List[Dict[str, Any]] = []
    texts_to_embed: List[str] = []
    chunk_refs: List[Dict[str, Any]] = []

    if message_data["body_plain"]:
        body_chunks = chunk_text(
            ChunkingInput(
                text=message_data["body_plain"],
                section_path="email:body",
                quoted_spans=quoted_spans,
                max_tokens=config.processing.chunk_size,
                overlap_tokens=config.processing.chunk_overlap,
            )
        )
        for c in body_chunks:
            c_dict: Dict[str, Any] = c.model_dump()
            c_dict["chunk_id"] = uuid.uuid4()
            c_dict["thread_id"] = thread_id
            c_dict["message_id"] = message_id
            c_dict["tenant_id"] = job.tenant_id

            chunks_data.append(c_dict)
            texts_to_embed.append(c.text)
            chunk_refs.append(c_dict)

    for att in attachments_data:
        if att["text"]:
            att_chunks = chunk_text(
                ChunkingInput(
                    text=att["text"],
                    section_path=f"attachment:{att['filename']}",
                    max_tokens=config.processing.chunk_size,
                    overlap_tokens=config.processing.chunk_overlap,
                )
            )
            for c in att_chunks:
                c_dict: Dict[str, Any] = c.model_dump()
                c_dict["chunk_id"] = uuid.uuid4()
                c_dict["thread_id"] = thread_id
                c_dict["message_id"] = message_id
                c_dict["attachment_id"] = att["attachment_id"]
                c_dict["tenant_id"] = job.tenant_id

                chunks_data.append(c_dict)
                texts_to_embed.append(c.text)
                chunk_refs.append(c_dict)

    if texts_to_embed:
        logger.info("Embedding %s chunks...", len(texts_to_embed))
        client = EmbeddingsClient()
        embeddings = client.embed_batch(texts_to_embed)

        for i, emb in enumerate(embeddings):
            chunk_refs[i]["embedding"] = emb
            chunk_refs[i][
                "embedding_model"
            ] = f"{config.embedding.model_name}:{config.embedding.output_dimensionality}"

    transformed_results: Dict[str, List[Dict[str, Any]]] = {
        "threads": [thread_data],
        "messages": [message_data],
        "attachments": attachments_data,
        "chunks": chunks_data,
    }

    with SessionLocal() as session:
        set_session_tenant(session, job.tenant_id)
        writer = DBWriter(session)
        writer.write_job_results(job, transformed_results)

    summary.messages_total = 1
    summary.messages_ingested = 1
    summary.attachments_total = len(attachments_data)
    summary.attachments_parsed = len(attachments_data)

    return summary
