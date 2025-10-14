#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import io
import json
import logging
import os
import re
import time
import hashlib
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np

try:
    from .index_metadata import (  # filenames + helpers (single source of truth)
        FAISS_INDEX_FILENAME,
        FILE_TIMES_FILENAME,
        INDEX_DIRNAME_DEFAULT,
        MAPPING_FILENAME,
        TIMESTAMP_FILENAME,
        create_index_metadata,
        index_paths,
        load_index_metadata,
        read_mapping,
        save_index_metadata,
        write_mapping,
    )
    # Try to import the consistency checker (preferred); fall back to a local checker below.
    try:
        from .index_metadata import check_index_consistency  # type: ignore
    except Exception:  # pragma: no cover
        check_index_consistency = None  # type: ignore
    from .llm_client import (
        embed_texts,  # shim over runtime (unit-normalized embeddings)
    )
    from .text_chunker import (
        prepare_index_units,  # emits id="doc_id::chunk{N}" when chunking
    )
    from .utils import (  # library-safe logger
        ensure_dir,
        find_conversation_dirs,
        load_conversation,
        logger,
        read_text_file,          # robust text reader (BOM/UTF-16/latin-1) + control-char sanitization
        clean_email_text,        # stronger cleaner for raw email bodies
    )
    from .config import EmailOpsConfig  # single source of truth for env/secrets + chunk defaults
except Exception:
    # Fallback for running as a script (no package context)
    import sys as _sys
    from pathlib import Path as _Path
    _pkg_dir = _Path(__file__).resolve().parent
    if str(_pkg_dir) not in _sys.path:
        _sys.path.insert(0, str(_pkg_dir))
    from index_metadata import (  # type: ignore
        FAISS_INDEX_FILENAME,
        FILE_TIMES_FILENAME,
        INDEX_DIRNAME_DEFAULT,
        MAPPING_FILENAME,
        TIMESTAMP_FILENAME,
        create_index_metadata,
        index_paths,
        load_index_metadata,
        read_mapping,
        save_index_metadata,
        write_mapping,
    )
    try:
        from index_metadata import check_index_consistency  # type: ignore
    except Exception:
        check_index_consistency = None  # type: ignore
    from llm_client import embed_texts  # type: ignore
    from text_chunker import prepare_index_units  # type: ignore
    from utils import (  # type: ignore
        ensure_dir, find_conversation_dirs, load_conversation, logger,
        read_text_file, clean_email_text
    )
    from config import EmailOpsConfig

try:
    import faiss  # type: ignore
    HAVE_FAISS = True
except Exception:
    faiss = None  # type: ignore
    HAVE_FAISS = False


# ============================================================================
# Constants
# ============================================================================

EMBED_MAX_BATCH = 250  # safe default for Vertex/HF/OpenAI et al.

# HIGH #13: File size limits for indexing
MAX_FILE_SIZE_MB = float(os.getenv("MAX_INDEXABLE_FILE_MB", "50"))
MAX_TEXT_CHARS = int(os.getenv("MAX_INDEXABLE_CHARS", "5000000"))

# File name constants
CONVERSATION_FILENAME = "Conversation.txt"
MANIFEST_FILENAME = "manifest.json"
SUMMARY_FILENAME = "summary.json"


# ============================================================================
# Atomic write helpers
# ============================================================================

def _atomic_write_bytes(dest: Path, data: bytes) -> None:
    """
    Write bytes atomically by writing to a temp file in the same directory and replacing.
    CRITICAL FIX #3: Added comprehensive error handling for disk full, permissions, and I/O errors.
    """
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    try:
        # Ensure parent directory exists
        dest.parent.mkdir(parents=True, exist_ok=True)
        
        # Write to temp file with error handling
        try:
            with open(tmp, "wb") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
        except OSError as e:
            # Handle disk full, permission errors, etc.
            if tmp.exists():
                with contextlib.suppress(Exception):
                    tmp.unlink()
            raise IOError(f"Failed to write temp file {tmp}: {e}") from e
        
        # Verify file was written correctly
        if not tmp.exists():
            raise IOError(f"Temp file {tmp} was not created")
        if tmp.stat().st_size != len(data):
            tmp.unlink()
            raise IOError(f"Temp file size mismatch: expected {len(data)}, got {tmp.stat().st_size}")
        
        # Atomic replace
        os.replace(tmp, dest)
        
        # Verify destination exists after replace
        if not dest.exists():
            raise IOError(f"Destination file {dest} does not exist after replace")
            
    except Exception as e:
        # Clean up temp file on any error
        if tmp.exists():
            with contextlib.suppress(Exception):
                tmp.unlink()
        raise IOError(f"Atomic write failed for {dest}: {e}") from e


def _atomic_write_text(dest: Path, text: str, *, encoding: str = "utf-8") -> None:
    """
    Write text atomically using UTF-8 by default.
    CRITICAL FIX #3: Added comprehensive error handling for disk full, permissions, and I/O errors.
    """
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    try:
        # Ensure parent directory exists
        dest.parent.mkdir(parents=True, exist_ok=True)
        
        # Write to temp file with error handling
        try:
            with open(tmp, "w", encoding=encoding) as f:
                f.write(text)
                f.flush()
                os.fsync(f.fileno())
        except OSError as e:
            # Handle disk full, permission errors, etc.
            if tmp.exists():
                with contextlib.suppress(Exception):
                    tmp.unlink()
            raise IOError(f"Failed to write temp file {tmp}: {e}") from e
        
        # Verify file was written correctly
        if not tmp.exists():
            raise IOError(f"Temp file {tmp} was not created")
        expected_size = len(text.encode(encoding))
        actual_size = tmp.stat().st_size
        # Allow some tolerance for encoding differences
        if abs(actual_size - expected_size) > expected_size * 0.1:
            tmp.unlink()
            raise IOError(f"Temp file size suspicious: expected ~{expected_size}, got {actual_size}")
        
        # Atomic replace
        os.replace(tmp, dest)
        
        # Verify destination exists after replace
        if not dest.exists():
            raise IOError(f"Destination file {dest} does not exist after replace")
            
    except Exception as e:
        # Clean up temp file on any error
        if tmp.exists():
            with contextlib.suppress(Exception):
                tmp.unlink()
        raise IOError(f"Atomic write failed for {dest}: {e}") from e


# ============================================================================
# GCP Credential Initialization (delegated to EmailOpsConfig)
# ============================================================================

def _initialize_gcp_credentials() -> None:
    """
    Keep a single source of truth for secrets/env wiring.
    """
    try:
        EmailOpsConfig.load().update_environment()
        logger.info("Initialized GCP credentials via EmailOpsConfig")
    except Exception as e:
        logger.error("Failed to initialize GCP credentials: %s", e)
        raise


# ============================================================================
# Small helpers
# ============================================================================

def _apply_model_override(provider: str, model: Optional[str]) -> None:
    """
    Map --model to the correct provider env var so llm_client picks it up.
    This only influences embedding calls in this process.
    """
    if not model:
        return
    env_map = {
        "vertex": "VERTEX_EMBED_MODEL",
        "openai": "OPENAI_EMBED_MODEL",
        "azure": "AZURE_OPENAI_DEPLOYMENT",  # deployment name
        "cohere": "COHERE_EMBED_MODEL",
        "huggingface": "HF_EMBED_MODEL",
        "qwen": "QWEN_EMBED_MODEL",
        "local": "LOCAL_EMBED_MODEL",
    }
    key = env_map.get((provider or "").lower())
    if key:
        os.environ[key] = str(model)


def _get_last_run_time(index_dir: Path) -> Optional[datetime]:
    p = index_dir / TIMESTAMP_FILENAME
    if not p.exists():
        return None
    try:
        txt = p.read_text(encoding="utf-8").strip()
        return datetime.fromisoformat(txt)
    except Exception:
        return None


def _save_run_time(index_dir: Path) -> None:
    # Atomic timestamp write
    _atomic_write_text(index_dir / TIMESTAMP_FILENAME, datetime.now(timezone.utc).isoformat(), encoding="utf-8")


def _clean_index_text(text: str) -> str:
    """
    Light cleaning to reduce noise for embeddings, without altering meaning.
    (Aggressive email cleanup happens upstream when needed.)
    """
    if not text:
        return ""
    s = re.sub(r"[ 	]+", " ", text)
    s = re.sub(r"\n{3,}", "\n\n", s)
    s = re.sub(r"[=\-_]{10,}", " ", s)
    s = re.sub(r"(?m)^-{20,}\n", "", s)
    return s.strip()


def _extract_manifest_metadata(conv: Dict[str, Any]) -> Dict[str, Any]:
    """Safe fields from manifest for indexing metadata."""
    man = conv.get("manifest") or {}
    subject = (man.get("smart_subject") or man.get("subject") or "").strip()

    # Participants (kept conservative; first message)
    participants: List[str] = []
    try:
        msgs = man.get("messages") or []
        if isinstance(msgs, list) and msgs:
            m0 = msgs[0] or {}
            if isinstance(m0, dict):
                if m0.get("from"):
                    f = m0["from"]
                    nm = (f.get("name") or f.get("smtp") or "").strip()
                    if nm:
                        participants.append(nm)
                for rec in (m0.get("to") or []):
                    if isinstance(rec, dict):
                        nm = (rec.get("name") or rec.get("smtp") or "").strip()
                        if nm:
                            participants.append(nm)
                for rec in (m0.get("cc") or []):
                    if isinstance(rec, dict):
                        nm = (rec.get("name") or rec.get("smtp") or "").strip()
                        if nm:
                            participants.append(nm)
    except Exception:
        pass

    # Dates
    start = None
    end = None
    try:
        span = man.get("time_span") or {}
        start = span.get("start_local") or span.get("start")
        end = span.get("end_local") or span.get("end")
    except Exception:
        pass

    return {
        "subject": subject,
        "participants": participants[:15],
        "start_date": start,
        "end_date": end,
    }


def _materialize_text_for_docs(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Ensure each doc has a non-empty 'text' field, using:
      1) existing 'text', else
      2) 'snippet' from prior mapping, else
      3) file content at 'path' (robustly read and sanitized)
    """
    out: List[Dict[str, Any]] = []
    for d in docs:
        t = str(d.get("text") or "")
        if not t.strip():
            snip = str(d.get("snippet") or "")
            if snip.strip():
                d["text"] = snip
            else:
                p = Path(str(d.get("path") or ""))
                if p.exists():
                    # Use robust reader (handles BOM/UTF-16/latin-1) and sanitizes control chars
                    try:
                        d["text"] = read_text_file(p)
                    except Exception:
                        d["text"] = ""
        out.append(d)
    return out


def _prefix_from_id(doc_id: str) -> str:
    """
    Normalize a document ID to its 2-part prefix for grouping purposes.
    
    Extracts the conversation-level identifier from chunk-level IDs:
      - "<conv_id>::conversation" (conversation main text)
      - "<conv_id>::att:{HASH}" (attachment)
      - "<conv_id>::conversation::chunk3" → "<conv_id>::conversation"
    
    Args:
        doc_id: Full document ID (may include ::chunk suffix)
        
    Returns:
        Two-part prefix without chunk suffix, or original ID if no '::' found
        
    Example:
        >>> _prefix_from_id("conv123::conversation::chunk2")
        'conv123::conversation'
        >>> _prefix_from_id("conv123::att:abc123")
        'conv123::att:abc123'
    """
    if not isinstance(doc_id, str):
        return ""
    parts = doc_id.split("::")
    return "::".join(parts[:2]) if len(parts) >= 2 else doc_id


def _iter_attachment_files(convo_dir: Path) -> Iterable[Path]:
    attachments_dir = convo_dir / "Attachments"
    if attachments_dir.exists():
        for p in attachments_dir.rglob("*"):
            if p.is_file():
                yield p
    # Also treat any file directly in convo_dir (except known files) as potential attachment
    for child in convo_dir.iterdir():
        if child.is_file() and child.name not in {CONVERSATION_FILENAME, MANIFEST_FILENAME, SUMMARY_FILENAME}:
            yield child


def _att_id(base_id: str, path: str) -> str:
    """
    Generate a stable attachment ID based on file path hash.
    
    Creates a consistent identifier for attachments using SHA-1 hash of
    the absolute POSIX path. This ensures the same attachment always gets
    the same ID across multiple index builds.
    
    Args:
        base_id: Base conversation ID (e.g., "conv123")
        path: File path to attachment (will be resolved to absolute)
        
    Returns:
        Stable attachment ID in format: "<base_id>::att:{sha1_hash[:12]}"
        
    Example:
        >>> _att_id("conv123", "/path/to/attachment.pdf")
        'conv123::att:a1b2c3d4e5f6'
    """
    try:
        ap = Path(path).resolve().as_posix()
    except Exception:
        ap = str(path)
    h = hashlib.sha1(ap.encode("utf-8")).hexdigest()[:12]
    return f"{base_id}::att:{h}"


# ============================================================================
# Chunk-building for a conversation (respects per-conversation limit)
# ============================================================================
def _build_doc_entries(
    conv: Dict[str, Any],
    convo_dir: Path,
    base_id: str,
    limit: Optional[int] = None,
    metadata: Optional[Dict[str, Any]] = None,
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> List[Dict[str, Any]]:
    """
    Build indexable document entries for a conversation and its attachments.

    Processes both the main Conversation.txt and any attachments, chunking
    text content and adding metadata. Enforces per-conversation limit and
    validates file sizes to prevent indexing oversized files.

    Returns:
        List of chunk dicts.
    """
    metadata = dict(metadata or {})
    out: List[Dict[str, Any]] = []

    # Conversation main text (use robust email cleaner)
    convo_txt_raw = conv.get("conversation_txt", "")

    # HIGH #13: Check file size before processing
    conv_file = convo_dir / CONVERSATION_FILENAME
    if conv_file.exists():
        size_mb = conv_file.stat().st_size / (1024 * 1024)
        if size_mb > MAX_FILE_SIZE_MB:
            logger.warning(
                "Skipping large conversation file: %s (%.1f MB > %.1f MB)",
                conv_file.name, size_mb, MAX_FILE_SIZE_MB
            )
            return out
        if len(convo_txt_raw) > MAX_TEXT_CHARS:
            logger.warning(
                "Truncating large conversation text: %s (%d chars > %d chars)",
                conv_file.name, len(convo_txt_raw), MAX_TEXT_CHARS
            )
            convo_txt_raw = convo_txt_raw[:MAX_TEXT_CHARS]

    convo_txt = clean_email_text(convo_txt_raw)

    # Stamp conversation mtime once (and fall back safely).
    try:
        conv_mtime = conv_file.stat().st_mtime if conv_file.exists() else time.time()
    except Exception:
        conv_mtime = time.time()

    if convo_txt:
        conv_id = f"{base_id}::conversation"
        conv_chunks = prepare_index_units(
            convo_txt,
            doc_id=conv_id,
            doc_path=str(convo_dir / CONVERSATION_FILENAME),
            subject=metadata.get("subject") or "",
            date=metadata.get("end_date") or metadata.get("start_date"),
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        upto = len(conv_chunks) if limit is None else max(0, min(limit, len(conv_chunks)))
        for ch in conv_chunks[:upto]:
            ch.update(
                {
                    "conv_id": base_id,
                    "doc_type": "conversation",
                    "subject": metadata.get("subject") or "",
                    "participants": metadata.get("participants") or [],
                    "start_date": metadata.get("start_date"),
                    "end_date": metadata.get("end_date"),
                    "modified_time": conv_mtime,
                }
            )
            out.append(ch)

    # Attachments (respect limit across attachments and chunks)
    for att in (conv.get("attachments") or []):
        if limit is not None and len(out) >= limit:
            break

        ap = Path(att.get("path", ""))

        # HIGH #13: Check attachment file size before processing
        if ap.exists():
            try:
                size_mb = ap.stat().st_size / (1024 * 1024)
            except OSError:
                # Treat unreadable stat as large: skip to be safe.
                logger.warning("Skipping unreadable attachment (stat failed): %s", ap)
                continue
            if size_mb > MAX_FILE_SIZE_MB:
                logger.warning(
                    "Skipping large attachment: %s (%.1f MB > %.1f MB)",
                    ap.name, size_mb, MAX_FILE_SIZE_MB
                )
                continue

        text_raw = att.get("text", "") or ""

        # HIGH #13: Check text length
        if len(text_raw) > MAX_TEXT_CHARS:
            logger.warning(
                "Truncating large attachment text: %s (%d chars > %d chars)",
                ap.name if ap.name else "<unnamed>", len(text_raw), MAX_TEXT_CHARS
            )
            text_raw = text_raw[:MAX_TEXT_CHARS]

        text = clean_email_text(text_raw)
        if not text.strip():
            continue

        att_id = _att_id(base_id, str(ap))
        att_chunks = prepare_index_units(
            text,
            doc_id=att_id,
            doc_path=str(ap),
            subject=metadata.get("subject") or (ap.name if ap.name else ""),
            date=metadata.get("end_date") or metadata.get("start_date"),
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        # Snapshot attachment mtime/size with robust fallbacks
        try:
            att_mtime = ap.stat().st_mtime if ap.exists() else time.time()
        except OSError:
            att_mtime = time.time()
        try:
            att_size = ap.stat().st_size if ap.exists() else None
        except OSError:
            att_size = None

        for ch in att_chunks:
            if limit is not None and len(out) >= limit:
                break
            ch.update(
                {
                    "conv_id": base_id,
                    "doc_type": "attachment",
                    "attachment_name": ap.name if ap.name else "",
                    "attachment_type": ap.suffix.lstrip(".").lower(),
                    "attachment_size": att_size,
                    "subject": metadata.get("subject") or "",
                    "participants": metadata.get("participants") or [],
                    "start_date": metadata.get("start_date"),
                    "end_date": metadata.get("end_date"),
                    "modified_time": att_mtime,
                }
            )
            out.append(ch)
    return out


# ============================================================================
# Corpus building (full or timestamp-based incremental)
# ============================================================================
def build_corpus(
    root: Path,
    index_dir: Path,
    *,
    last_run_time: Optional[datetime] = None,
    limit: Optional[int] = None,
    chunk_size: int,
    chunk_overlap: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Full scan of conversation folders; when last_run_time is provided, reuses prior
    mapping to keep truly unchanged docs (including conversation unchanged + attachments unchanged).
    Returns (new_or_updated_docs, unchanged_docs).
    """
    new_or_updated_docs: List[Dict[str, Any]] = []
    unchanged_docs: List[Dict[str, Any]] = []

    conversation_dirs = find_conversation_dirs(root)  # list of Path to conversation folder
    old_mapping_list = read_mapping(index_dir)
    old_mapping: Dict[str, Dict[str, Any]] = {
        str(d.get("id")): d for d in (old_mapping_list or []) if d.get("id") is not None
    }

    for convo_dir in conversation_dirs:
        base_id = convo_dir.name
        convo_txt_path = convo_dir / CONVERSATION_FILENAME
        if not convo_txt_path.exists():
            continue

        # When last_run_time is provided, only do the "unchanged conversation" fast-path
        # if we actually have prior docs for this conversation. Otherwise, treat as new.
        if last_run_time is not None:
            try:
                modified_time = datetime.fromtimestamp(convo_txt_path.stat().st_mtime, tz=timezone.utc)
            except Exception:
                modified_time = None

            # Inventory old docs for this conversation
            convo_prefix = f"{base_id}::"
            prior_docs_for_conv = [
                doc for did, doc in old_mapping.items()
                if isinstance(did, str) and did.startswith(convo_prefix)
            ]

            if modified_time and modified_time < last_run_time and prior_docs_for_conv:
                # Conversation main text unchanged — check attachments
                conv = load_conversation(convo_dir)
                meta = _extract_manifest_metadata(conv)

                # Detect changed attachments (>= last run or stat() fails)
                changed_attachment_paths: Set[Path] = set()
                last_ts = last_run_time.timestamp()
                for att_file in _iter_attachment_files(convo_dir):
                    try:
                        if att_file.stat().st_mtime >= last_ts:
                            changed_attachment_paths.add(att_file.resolve())
                    except OSError:
                        # Consider unreadable stat as changed to be safe
                        changed_attachment_paths.add(att_file.resolve())

                # Build mapping path->stable att_id for changed ones
                path_to_attid: Dict[Path, str] = {}
                for att in (conv.get("attachments") or []):
                    try:
                        path_to_attid[Path(att.get("path", "")).resolve()] = _att_id(base_id, str(att.get("path", "")))
                    except Exception:
                        continue

                # Compute prefixes that changed due to mtime
                changed_prefixes: Set[str] = set()
                for p in changed_attachment_paths:
                    aid = path_to_attid.get(p)
                    if aid:
                        changed_prefixes.add(aid)

                # Existing (current) attachment prefixes
                current_att_prefixes: Set[str] = {
                    _att_id(base_id, str(att.get("path", "")))
                    for att in (conv.get("attachments") or [])
                }
                # Previously existing attachment prefixes for this conversation
                old_att_prefixes: Set[str] = {
                    _prefix_from_id(d.get("id", ""))
                    for d in prior_docs_for_conv
                    if str(d.get("id", "")).startswith(f"{base_id}::att")
                }
                # Any newly-added attachments (present now but not before) must be rebuilt
                for pref in current_att_prefixes - old_att_prefixes:
                    changed_prefixes.add(pref)

                # Keep only truly unchanged docs (not changed AND still present)
                for d in prior_docs_for_conv:
                    did = d.get("id", "")
                    prefix = _prefix_from_id(did)
                    # drop if attachment removed
                    if prefix.startswith(f"{base_id}::att") and prefix not in current_att_prefixes:
                        continue
                    # drop if attachment changed
                    if any(did == pref or did.startswith(pref + "::") for pref in changed_prefixes):
                        continue
                    unchanged_docs.append(d)

                # Rebuild changed attachments (respect per-conversation limit)
                added_for_conv = 0
                for att in (conv.get("attachments") or []):
                    ap = Path(att.get("path", ""))
                    pref = _att_id(base_id, str(ap))
                    if pref not in changed_prefixes:
                        continue
                    if not (att.get("text") or "").strip():
                        continue
                    att_meta = dict(meta)
                    att_meta["subject"] = att_meta.get("subject") or ap.name
                    chunks = prepare_index_units(
                        clean_email_text(att.get("text", "")),
                        doc_id=pref,
                        doc_path=str(ap),
                        subject=att_meta.get("subject") or "",
                        date=att_meta.get("end_date") or att_meta.get("start_date"),
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                    )
                    try:
                        mt = ap.stat().st_mtime
                    except Exception:
                        mt = time.time()
                    for ch in chunks:
                        if limit is not None and added_for_conv >= limit:
                            break
                        ch.update(
                            {
                                "conv_id": base_id,
                                "doc_type": "attachment",
                                "attachment_name": ap.name,
                                "attachment_type": ap.suffix.lstrip(".").lower(),
                                "attachment_size": ap.stat().st_size if ap.exists() else None,
                                "subject": att_meta.get("subject") or "",
                                "participants": att_meta.get("participants") or [],
                                "start_date": att_meta.get("start_date"),
                                "end_date": att_meta.get("end_date"),
                            }
                        )
                        ch["modified_time"] = mt
                        new_or_updated_docs.append(ch)
                        added_for_conv += 1
                continue  # next conversation

        # If conversation changed or no last_run_time -> rebuild fully
        conv = load_conversation(convo_dir)
        meta = _extract_manifest_metadata(conv)
        docs = _build_doc_entries(conv, convo_dir, base_id, limit=limit, metadata=meta,
                                  chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        try:
            mt = convo_txt_path.stat().st_mtime
        except Exception:
            mt = time.time()
        for d in docs:
            if "modified_time" not in d or d.get("modified_time") is None:
                d["modified_time"] = mt
        new_or_updated_docs.extend(docs)

    return new_or_updated_docs, unchanged_docs


# ============================================================================
# File-times incremental (precise; supports deletions correctly)
# ============================================================================
def build_incremental_corpus(
    root: Path,
    existing_file_times: Dict[str, float],
    existing_mapping: List[Dict[str, Any]],
    limit: Optional[int] = None,
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> Tuple[List[Dict[str, Any]], Set[str]]:
    """
    True incremental build using stored per-doc file times.
    Returns (new_or_updated_docs, deleted_doc_ids).

    Fixes:
      * Track deletions at the *chunk-level* by comparing existing mapping to current folder state.
      * Remove old chunks for changed attachments and conversations to avoid duplication.
      * Enforce per-conversation `limit` for newly rebuilt records.
    """
    new_docs: List[Dict[str, Any]] = []
    deleted_ids: Set[str] = set()

    # Group prior mapping rows by conversation and by 2-part prefix
    by_conv: DefaultDict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in existing_mapping or []:
        cid = str(row.get("conv_id") or "")
        if cid:
            by_conv[cid].append(row)

    by_conv_prefix: Dict[str, Dict[str, List[str]]] = {}
    for cid, rows in by_conv.items():
        pref_map: DefaultDict[str, List[str]] = defaultdict(list)
        for r in rows:
            rid = str(r.get("id") or "")
            pref_map[_prefix_from_id(rid)].append(rid)
        by_conv_prefix[cid] = dict(pref_map)

    # Current conversations present on disk
    current_dirs = find_conversation_dirs(root)
    current_conv_ids: Set[str] = {p.name for p in current_dirs}

    for convo_dir in current_dirs:
        base_id = convo_dir.name
        conv = load_conversation(convo_dir)
        meta = _extract_manifest_metadata(conv)

        # --- Conversation change detection ---
        conv_doc_id = f"{base_id}::conversation"
        try:
            conv_mtime = (convo_dir / CONVERSATION_FILENAME).stat().st_mtime
        except Exception:
            try:
                conv_mtime = convo_dir.stat().st_mtime
            except Exception:
                conv_mtime = time.time()

        prior_times = [t for did, t in existing_file_times.items() if did == conv_doc_id or did.startswith(conv_doc_id + "::")]
        prior_time = max(prior_times) if prior_times else None
        added_for_conv = 0

        if (prior_time or 0) < conv_mtime:
            # Rebuild conversation
            txt = clean_email_text(conv.get("conversation_txt", ""))
            chunks = prepare_index_units(
                txt,
                doc_id=conv_doc_id,
                doc_path=str(convo_dir / CONVERSATION_FILENAME),
                subject=meta.get("subject") or "",
                date=meta.get("end_date") or meta.get("start_date"),
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            # Old conv chunks → delete to avoid duplication
            for rid in (by_conv_prefix.get(base_id, {}).get(conv_doc_id, []) or []):
                deleted_ids.add(rid)

            for ch in chunks:
                if limit is not None and added_for_conv >= limit:
                    break
                ch.update(
                    {
                        "conv_id": base_id,
                        "doc_type": "conversation",
                        "subject": meta.get("subject") or "",
                        "participants": meta.get("participants") or [],
                        "start_date": meta.get("start_date"),
                        "end_date": meta.get("end_date"),
                        "modified_time": conv_mtime,
                    }
                )
                new_docs.append(ch)
                added_for_conv += 1

        # --- Attachments change / deletion detection ---
        # Map current attachments (with stable IDs)
        current_att_prefixes: Set[str] = set()
        for att in (conv.get("attachments") or []):
            ap = Path(att.get("path", ""))
            prefix = _att_id(base_id, str(ap))
            current_att_prefixes.add(prefix)
            try:
                mt = ap.stat().st_mtime
            except Exception:
                mt = time.time()

            prior_times = [t for did, t in existing_file_times.items() if did == prefix or did.startswith(prefix + "::")]
            prior_time = max(prior_times) if prior_times else None

            if (prior_time or 0) < mt and (att.get("text") or "").strip():
                # Delete old chunks for this attachment prefix to avoid duplication
                for rid in (by_conv_prefix.get(base_id, {}).get(prefix, []) or []):
                    deleted_ids.add(rid)

                text = clean_email_text(att["text"])
                chunks = prepare_index_units(
                    text,
                    doc_id=prefix,
                    doc_path=str(ap),
                    subject=meta.get("subject") or ap.name,
                    date=meta.get("end_date") or meta.get("start_date"),
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )
                for ch in chunks:
                    if limit is not None and added_for_conv >= limit:
                        break
                    try:
                        _att_size = ap.stat().st_size if ap.exists() else None
                    except OSError:
                        _att_size = None
                    ch.update(
                        {
                            "conv_id": base_id,
                            "doc_type": "attachment",
                            "attachment_name": ap.name,
                            "attachment_type": ap.suffix.lstrip(".").lower(),
                            "attachment_size": _att_size,
                            "subject": meta.get("subject") or "",
                            "participants": meta.get("participants") or [],
                            "start_date": meta.get("start_date"),
                            "end_date": meta.get("end_date"),
                            "modified_time": mt,
                        }
                    )
                    new_docs.append(ch)
                    added_for_conv += 1

        # Any attachment prefixes that existed before but are now missing → delete
        prior_prefixes = set((by_conv_prefix.get(base_id) or {}).keys())
        missing_att_prefixes = {
            p for p in prior_prefixes
            if p.startswith(f"{base_id}::att") and p not in current_att_prefixes
        }
        for mpref in missing_att_prefixes:
            for rid in by_conv_prefix.get(base_id, {}).get(mpref, []) or []:
                deleted_ids.add(rid)

    # Conversations that no longer exist on disk → delete all their rows
    for old_cid in set(by_conv.keys()) - current_conv_ids:
        for row in by_conv.get(old_cid, []):
            rid = str(row.get("id") or "")
            if rid:
                deleted_ids.add(rid)

    return new_docs, deleted_ids


# ============================================================================
# Index I/O
# ============================================================================
def load_existing_index(index_dir: Path) -> Tuple[Optional[Any], Optional[List[Dict[str, Any]]], Optional[Dict[str, float]], Optional[np.ndarray]]:
    """
    Returns (faiss_index_or_None, mapping_list_or_None, file_times_dict_or_None, embeddings_array_or_None)
    All parts are optional; function handles missing files gracefully.
    """
    ixp = index_paths(index_dir)

    mapping = read_mapping(index_dir)

    file_times: Optional[Dict[str, float]] = None
    if ixp.file_times.exists():
        try:
            file_times = json.loads(ixp.file_times.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning("Failed to read %s: %s", ixp.file_times.name, e)

    embeddings: Optional[np.ndarray] = None
    if ixp.embeddings.exists():
        try:
            embeddings = np.load(str(ixp.embeddings), mmap_mode="r").astype("float32", copy=False)
        except Exception as e:
            logger.warning("Failed to read %s: %s", ixp.embeddings.name, e)

    fidx = None
    if HAVE_FAISS and faiss is not None and ixp.faiss.exists():
        try:
            fidx = faiss.read_index(str(ixp.faiss))  # type: ignore
        except Exception as e:
            logger.warning("Failed to load FAISS index: %s", e)

    return fidx, mapping, file_times, embeddings


def _local_check_index_consistency(index_dir: Path) -> None:
    """Fallback consistency check when package-provided checker is unavailable."""
    try:
        ixp = index_paths(index_dir)
        # Embeddings vs mapping count
        m = read_mapping(index_dir) or []
        emb = None
        if ixp.embeddings.exists():
            try:
                emb = np.load(str(ixp.embeddings), mmap_mode="r")
            except Exception:
                emb = None
        if emb is not None and emb.shape[0] != len(m):
            raise RuntimeError(f"Embeddings/document count mismatch: {emb.shape[0]} vs {len(m)}")
        # FAISS ntotal vs embeddings
        if HAVE_FAISS and faiss is not None and ixp.faiss.exists() and emb is not None:
            try:
                fidx = faiss.read_index(str(ixp.faiss))  # type: ignore
                if int(getattr(fidx, 'ntotal', 0)) != int(emb.shape[0]):
                    raise RuntimeError(f"FAISS ntotal {getattr(fidx, 'ntotal', 0)} != embeddings {emb.shape[0]}")
            except Exception as e:
                raise RuntimeError(f"FAISS consistency check failed: {e}")
    except Exception as e:
        raise


def save_index(index_dir: Path, embeddings: np.ndarray, mapping: List[Dict[str, Any]], *, provider: str, num_folders: int) -> None:
    """
    Persist embeddings (embeddings.npy), mapping.json, (optional) FAISS index, and meta.json.
    All writes are atomic. A post-save consistency check verifies alignment.
    """
    ixp = index_paths(index_dir)
    ensure_dir(ixp.base)

    # 1) embeddings (atomic NPY write via BytesIO buffer -> bytes)
    buf = io.BytesIO()
    np.save(buf, embeddings.astype("float32", order="C"))
    _atomic_write_bytes(ixp.embeddings, buf.getvalue())

    # 2) mapping (writer is already atomic)
    write_mapping(index_dir, mapping)

    # 3) optional FAISS (Inner Product index - cosine ready because vectors are unit-normalized)
    if HAVE_FAISS and faiss is not None:
        try:
            dim = int(embeddings.shape[1])
            index = faiss.IndexFlatIP(dim)  # type: ignore
            index.add(np.ascontiguousarray(embeddings, dtype=np.float32))  # type: ignore
            # Write FAISS to a temp path then replace
            faiss_tmp = ixp.faiss.with_suffix(ixp.faiss.suffix + ".tmp")
            faiss.write_index(index, str(faiss_tmp))  # type: ignore
            os.replace(faiss_tmp, ixp.faiss)
        except Exception as e:
            logger.warning("FAISS indexing failed, continuing without faiss: %s", e)

    # 4) metadata
    meta = create_index_metadata(
        provider=provider,
        num_documents=len(mapping),
        num_folders=int(num_folders),
        index_dir=index_dir,
        custom_metadata={"actual_dimensions": int(embeddings.shape[1])},
    )
    save_index_metadata(meta, index_dir)
    logger.info("Saved index (vectors=%d, dimensions=%d)", embeddings.shape[0], embeddings.shape[1])

    # 5) Post-save consistency check
    try:
        if check_index_consistency is not None:
            check_index_consistency(index_dir, raise_on_mismatch=True)  # type: ignore
        else:
            _local_check_index_consistency(index_dir)
    except Exception as e:
        logger.error("Post-save consistency check failed: %s", e)
        raise


# ============================================================================
# CLI
# ============================================================================
def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    ap = argparse.ArgumentParser(description="Build or update the email search index.")
    ap.add_argument("--root", required=True, help="Export root that contains conversation folders and (optionally) an _index/ directory")
    # Constrain provider to 'vertex' for this build to match metadata/validator support.
    ap.add_argument("--provider", choices=["vertex"], default=os.getenv("EMBED_PROVIDER", "vertex"), help="Embedding provider for index build (this build supports only 'vertex')")
    ap.add_argument("--model", help="Embedding model/deployment override for the chosen provider")
    ap.add_argument("--batch", type=int, default=int(os.getenv("EMBED_BATCH", "64")), help="Embedding batch size (e.g., 64-250 for Vertex, will be clamped to max 250)")
    ap.add_argument("--index-root", help="Directory where the _index folder should be created (defaults to --root)")
    ap.add_argument("--force-reindex", action="store_true", help="Force a full re-index of all conversations")
    ap.add_argument("--limit", type=int, help="Limit number of chunks per conversation (for quick smoke tests)")
    args = ap.parse_args()

    # Initialize GCP credentials for Vertex AI via config (single source of truth).
    if args.provider == "vertex":
        _initialize_gcp_credentials()

    root = Path(args.root).expanduser().resolve()
    index_base = Path(args.index_root).expanduser().resolve() if args.index_root else root

    if not index_base.exists():
        logger.info("Index base '%s' does not exist yet; creating it", index_base)
        index_base.mkdir(parents=True, exist_ok=True)

    out_dir = index_base / INDEX_DIRNAME_DEFAULT

    ensure_dir(out_dir)

    # Optional: pin model via env
    _apply_model_override(args.provider, args.model)

    # Use config-backed chunk parameters (threaded to all prepare_index_units calls)
    try:
        cfg = EmailOpsConfig.load()
        chunk_size = int(getattr(cfg, "DEFAULT_CHUNK_SIZE", 1600))
        chunk_overlap = int(getattr(cfg, "DEFAULT_CHUNK_OVERLAP", 200))
    except Exception:
        chunk_size, chunk_overlap = 1600, 200  # safe defaults matching config.py

    # Load previous index bits (if any)
    _, existing_mapping, existing_file_times, existing_embeddings = load_existing_index(out_dir)
    existing_meta = load_index_metadata(out_dir)  # may be None

    # Last-run strategy
    last_run_time = None if args.force_reindex else _get_last_run_time(out_dir)

    # Provider resolution (prefer index provider to avoid dim mismatches)
    embed_provider = args.provider
    if not args.force_reindex and existing_embeddings is not None and existing_mapping:
        idx_provider = (existing_meta.get("provider") or args.provider) if existing_meta else args.provider
        if idx_provider and idx_provider != args.provider:
            logger.warning(
                "Using index provider '%s' instead of requested '%s' to ensure dimensional compatibility",
                idx_provider, args.provider
            )
        embed_provider = idx_provider

    # ------------------------------------------------------------------
    # 1) Build doc corpus (new/updated + unchanged)
    # ------------------------------------------------------------------
    if existing_file_times and not args.force_reindex:
        # precise incremental with correct deletions
        new_docs, deleted_ids = build_incremental_corpus(
            root,
            existing_file_times,
            existing_mapping or [],
            limit=args.limit,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        new_ids = {d["id"] for d in new_docs}
        unchanged_docs = [
            d for d in (existing_mapping or [])
            if d.get("id") not in new_ids and d.get("id") not in deleted_ids
        ]
        logger.info(
            "Incremental corpus: %d new/updated, %d unchanged, %d deleted",
            len(new_docs), len(unchanged_docs), len(deleted_ids)
        )
    else:
        # timestamp/manifest based
        if last_run_time:
            logger.info(
                "Starting incremental (timestamp) update from %s",
                last_run_time.strftime("%Y-%m-%d %H:%M:%S %Z")
            )
        new_docs, unchanged_docs = build_corpus(root, out_dir, last_run_time=last_run_time, limit=args.limit,
                                                chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        logger.info("Corpus: %d new/updated, %d unchanged", len(new_docs), len(unchanged_docs))

    # Short-circuit: nothing to do
    if not new_docs and existing_embeddings is not None and existing_mapping and unchanged_docs and not args.force_reindex:
        logger.info("No new content; index remains unchanged.")
        _save_run_time(out_dir)
        return

    # ------------------------------------------------------------------
    # 2) Embed texts (reuse unchanged vectors when possible)
    # ------------------------------------------------------------------
    final_docs: List[Dict[str, Any]] = []
    all_embeddings: List[np.ndarray] = []

    if not (new_docs or unchanged_docs):
        logger.info("No documents to index.")
        _save_run_time(out_dir)
        return

    # Clamp batch size safely
    batch = max(1, min(int(args.batch or 64), EMBED_MAX_BATCH))

    def _validate_batch(vecs: np.ndarray, expected_rows: int) -> None:
        if vecs.size == 0:
            raise RuntimeError("Embedding provider returned empty vectors; check provider credentials and model.")
        if vecs.ndim != 2 or vecs.shape[0] != int(expected_rows):
            raise RuntimeError(f"Invalid embeddings shape: got {vecs.shape}, expected rows={expected_rows}")
        if not np.isfinite(vecs).all():
            raise RuntimeError("Invalid embeddings returned (non-finite values detected)")
        # At least one row must have a reasonable norm (~1.0 for unit-normalized embeddings)
        if float(np.max(np.linalg.norm(vecs, axis=1))) < 1e-3:
            raise RuntimeError("Embeddings look degenerate (all ~zero)")

    if existing_embeddings is not None and existing_mapping and not args.force_reindex:
        # Reuse unchanged vectors by id -> row index
        id_to_old_idx = {doc["id"]: i for i, doc in enumerate(existing_mapping)}
        unchanged_with_vecs: List[Dict[str, Any]] = []
        unchanged_to_embed: List[Dict[str, Any]] = []

        for d in unchanged_docs:
            idx = id_to_old_idx.get(d.get("id"))
            if idx is not None and 0 <= idx < existing_embeddings.shape[0]:
                unchanged_with_vecs.append(d)
                all_embeddings.append(existing_embeddings[idx: idx + 1])  # view slice
            else:
                unchanged_to_embed.append(d)

        # Prepare texts for embedding (fill from snippet or file path if needed)
        docs_to_embed = new_docs + unchanged_to_embed
        docs_to_embed = _materialize_text_for_docs(docs_to_embed)
        valid_docs = [d for d in docs_to_embed if str(d.get("text", "")).strip()]
        texts = [str(d["text"]) for d in valid_docs]

        if valid_docs:
            for i in range(0, len(texts), batch):
                chunk = texts[i: i + batch]
                vecs = embed_texts(chunk, provider=embed_provider)  # normalized vectors
                vecs = np.asarray(vecs, dtype="float32")
                _validate_batch(vecs, expected_rows=len(chunk))
                all_embeddings.append(vecs)

        final_docs = unchanged_with_vecs + valid_docs
    else:
        # Full embedding (fresh index or no reusable vectors)
        docs = list(new_docs + unchanged_docs)
        docs = _materialize_text_for_docs(docs)
        docs = [d for d in docs if str(d.get("text", "")).strip()]
        if not docs:
            logger.info("No non-empty documents to embed.")
            _save_run_time(out_dir)
            return

        texts = [str(d["text"]) for d in docs]
        for i in range(0, len(texts), batch):
            chunk = texts[i: i + batch]
            vecs = embed_texts(chunk, provider=embed_provider)
            vecs = np.asarray(vecs, dtype="float32")
            _validate_batch(vecs, expected_rows=len(chunk))
            all_embeddings.append(vecs)
        final_docs = docs

    if not all_embeddings:
        logger.info("Nothing to embed after filtering; aborting update.")
        _save_run_time(out_dir)
        return

    embeddings = np.vstack(all_embeddings)
    if embeddings.shape[0] != len(final_docs):
        raise RuntimeError(f"Embeddings/document count mismatch: {embeddings.shape[0]} vectors for {len(final_docs)} docs")

    # ------------------------------------------------------------------
    # 3) Persist index artifacts
    # ------------------------------------------------------------------
    mapping_out: List[Dict[str, Any]] = []
    file_times: Dict[str, float] = {}
    # capture prior file_times to preserve unchanged values
    prior_times = existing_file_times or {}

    for d in final_docs:
        text = str(d.get("text", "") or "")
        snippet = text[:500] if text else str(d.get("snippet", "") or "")[:500]
        
        # Compute content hash for deduplication (SHA-256 of cleaned text)
        # This enables efficient duplicate detection at search time
        content_hash = ""
        if text:
            try:
                content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]  # First 64 bits
            except Exception:
                content_hash = ""
        
        rec = {
            "id": d.get("id"),
            "path": d.get("path"),
            "conv_id": d.get("conv_id"),
            "doc_type": d.get("doc_type"),
            "subject": d.get("subject", ""),
            "date": d.get("date") or d.get("end_date") or d.get("start_date"),
            "start_date": d.get("start_date"),
            "end_date": d.get("end_date"),
            "from_email": d.get("from_email", ""),
            "from_name": d.get("from_name", ""),
            "to_emails": d.get("to_emails", []),
            "cc_emails": d.get("cc_emails", []),
            "participants": d.get("participants", []),
            "attachment_name": d.get("attachment_name"),
            "attachment_type": d.get("attachment_type"),
            "attachment_size": d.get("attachment_size"),
            "snippet": snippet,
            "content_hash": content_hash,  # NEW: Enable efficient deduplication
        }
        mapping_out.append(rec)

        # Preserve previous modified_time for unchanged docs when missing
        doc_id = str(d.get("id"))
        mt = d.get("modified_time", None)
        if mt is None:
            mt = prior_times.get(doc_id, None)
        file_times[doc_id] = float(mt if mt is not None else time.time())

    save_index(out_dir, embeddings, mapping_out, provider=embed_provider, num_folders=len({(d["id"] or "").split("::")[0] for d in final_docs}))

    # file_times.json (atomic) + timestamp (atomic)
    ixp = index_paths(out_dir)
    try:
        _atomic_write_text(ixp.file_times, json.dumps(file_times, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        logger.warning("Failed to write %s: %s", ixp.file_times.name, e)

    _save_run_time(out_dir)
    logger.info("Index updated at %s", out_dir)


if __name__ == "__main__":
    main()
