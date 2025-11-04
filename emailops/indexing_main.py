#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import time
from collections import defaultdict
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

try:
    from .indexing_metadata import (  # filenames + helpers (single source of truth)
        INDEX_DIRNAME_DEFAULT,
        TIMESTAMP_FILENAME,
        create_index_metadata,
        index_paths,
        load_index_metadata,
        read_mapping,
    )

    # Try to import the consistency checker (preferred); fall back to a local checker below.
    try:
        from .indexing_metadata import check_index_consistency  # type: ignore
    except Exception:  # pragma: no cover
        check_index_consistency = None  # type: ignore
    from .core_config import EmailOpsConfig
    from .core_conversation_loader import load_conversation
    from .core_email_processing import clean_email_text
    from .core_manifest import extract_metadata_lightweight  # CENTRALIZED
    from .index_transaction import IndexTransaction, recover_index_from_wal  # P0-4 FIX
    from .llm_client_shim import embed_texts
    from .llm_text_chunker import prepare_index_units
    from .services.file_service import FileService
    from .utils import (
        ensure_dir,
        find_conversation_dirs,
        logger,
    )
except Exception:
    # Fallback for running as a script (no package context)
    import sys as _sys
    from pathlib import Path as _Path

    _pkg_dir = _Path(__file__).resolve().parent
    if str(_pkg_dir) not in _sys.path:
        _sys.path.insert(0, str(_pkg_dir))
    from indexing_metadata import (  # type: ignore
        INDEX_DIRNAME_DEFAULT,
        TIMESTAMP_FILENAME,
        create_index_metadata,
        index_paths,
        load_index_metadata,
        read_mapping,
    )

    try:
        from indexing_metadata import check_index_consistency  # type: ignore
    except Exception:
        check_index_consistency = None  # type: ignore
    from core_config import EmailOpsConfig  # type: ignore
    from core_conversation_loader import load_conversation  # type: ignore
    from core_email_processing import clean_email_text  # type: ignore
    from core_manifest import extract_metadata_lightweight  # type: ignore - CENTRALIZED
    from index_transaction import (  # type: ignore - P0-4 FIX
        IndexTransaction,
        recover_index_from_wal,
    )
    from llm_client_shim import embed_texts  # type: ignore
    from llm_text_chunker import prepare_index_units  # type: ignore
    from services.file_service import FileService
    from utils import (  # type: ignore
        ensure_dir,
        find_conversation_dirs,
        logger,
    )

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



# ----------------------------------------------------------------------------
# Limits helper (prefer centralized config, fall back to env constants)
# ----------------------------------------------------------------------------
def _get_index_limits() -> tuple[float, int]:
    try:
        cfg = EmailOpsConfig.load()
        return float(cfg.limits.max_indexable_file_mb), int(cfg.limits.max_indexable_chars)
    except Exception:
        return MAX_FILE_SIZE_MB, MAX_TEXT_CHARS


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


def _apply_model_override(provider: str, model: str | None) -> None:
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


def _get_last_run_time(index_dir: Path) -> datetime | None:
    p = index_dir / TIMESTAMP_FILENAME
    if not p.exists():
        return None
    try:
        txt = p.read_text(encoding="utf-8").strip()
        return datetime.fromisoformat(txt)
    except Exception as e:
        logger.debug("Failed to parse timestamp: %s", e)
        return None


def _save_run_time(index_dir: Path) -> None:
    # P2-6 FIX: Use FileService for atomic writes
    file_service = FileService(export_root=str(index_dir.parent))
    file_service.save_text_file(
        datetime.now(UTC).isoformat(),
        index_dir / TIMESTAMP_FILENAME,
    )


def _clean_index_text(text: str) -> str:
    """
    Light cleaning to reduce noise for embeddings, without altering meaning.
    (Aggressive email cleanup happens upstream when needed.)

    LOW #44: Function name is accurate - performs light cleaning optimized for indexing.
    The "light" qualifier is intentional to distinguish from aggressive email cleaning.
    """
    if not text:
        return ""
    s = re.sub(r"[ 	]+", " ", text)
    s = re.sub(r"\n{3,}", "\n\n", s)
    s = re.sub(r"[=\-_]{10,}", " ", s)
    s = re.sub(r"(?m)^-{20,}\n", "", s)
    return s.strip()




def _materialize_text_for_docs(docs: list[dict[str, Any]], file_service: FileService) -> list[dict[str, Any]]:
    """
    Ensure each doc has a non-empty 'text' field, using:
      1) existing 'text', else
      2) 'snippet' from prior mapping, else
      3) file content at 'path' (robustly read and sanitized)
    """
    out: list[dict[str, Any]] = []
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
                        d["text"] = file_service.read_text_file(p)
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
        if child.is_file() and child.name not in {
            CONVERSATION_FILENAME,
            MANIFEST_FILENAME,
            SUMMARY_FILENAME,
        }:
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
    h = hashlib.sha256(ap.encode("utf-8"), usedforsecurity=False).hexdigest()[:12]
    return f"{base_id}::att:{h}"


# ============================================================================
# Chunk-building for a conversation (respects per-conversation limit)
# ============================================================================
def _load_pre_chunked_data(chunk_file: Path, base_id: str, conv_mtime: float, limit: int | None) -> list[dict[str, Any]]:
    """Load pre-chunked data from a file."""
    out = []
    try:
        chunks_data = json.loads(chunk_file.read_text(encoding="utf-8"))
        if isinstance(chunks_data, list):
            logger.info(
                "✓ Loaded %d pre-chunked items for %s (already cleaned)",
                len(chunks_data),
                base_id,
            )
            upto = len(chunks_data) if limit is None else max(0, min(limit, len(chunks_data)))
            for ch in chunks_data[:upto]:
                if isinstance(ch, dict):
                    ch.update(
                        {
                            "conv_id": base_id,
                            "doc_type": ch.get("doc_type", "conversation"),
                            "modified_time": conv_mtime,
                        }
                    )
                    out.append(ch)
    except Exception as e:
        logger.warning(
            "Failed to load chunk file %s: %s, falling back to fresh clean+chunk",
            chunk_file,
            e,
        )
        return []
    return out

def _build_doc_entries(
    convo_dir: Path,
    base_id: str,
    limit: int | None = None,
    metadata: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """
    Build indexable document entries for a conversation and its attachments.
    """
    metadata = dict(metadata or {})
    chunks_dir = convo_dir.parent / "_chunks"
    chunk_file = chunks_dir / f"{base_id}.json"
    conv_mtime = time.time()
    try:
        conv_file = convo_dir / CONVERSATION_FILENAME
        conv_mtime = conv_file.stat().st_mtime if conv_file.exists() else time.time()
    except Exception as e:
        logger.debug("Failed to get file mtime, using current time: %s", e)

    if chunk_file.exists():
        pre_chunked = _load_pre_chunked_data(chunk_file, base_id, conv_mtime, limit)
        if pre_chunked:
            return pre_chunked

    logger.debug("No valid chunk file for %s, performing fresh chunking", base_id)
    # Fresh chunking fallback - return empty list if no chunks can be created
    return []


# ============================================================================
# Corpus building (full or timestamp-based incremental)
# ============================================================================
def build_corpus(
    root: Path,
    index_dir: Path,
    *,
    last_run_time: datetime | None = None,
    limit: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Full scan of conversation folders
    when last_run_time is provided, reuses prior
    mapping to keep truly unchanged docs (including conversation unchanged + attachments unchanged).
    Returns (new_or_updated_docs, unchanged_docs).
    """
    new_or_updated_docs: list[dict[str, Any]] = []
    unchanged_docs: list[dict[str, Any]] = []

    conversation_dirs = find_conversation_dirs(
        root
    )  # list of Path to conversation folder
    old_mapping_list = read_mapping(index_dir)
    old_mapping: dict[str, dict[str, Any]] = {
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
                modified_time = datetime.fromtimestamp(
                    convo_txt_path.stat().st_mtime, tz=UTC
                )
            except Exception:
                modified_time = None

            # Inventory old docs for this conversation
            convo_prefix = f"{base_id}::"
            prior_docs_for_conv = [
                doc
                for did, doc in old_mapping.items()
                if isinstance(did, str) and did.startswith(convo_prefix)
            ]

            if modified_time and modified_time < last_run_time and prior_docs_for_conv:
                # Conversation main text unchanged - check attachments
                # HIGH #38: Use consistent load_conversation parameters
                conv = load_conversation(
                    convo_dir,
                    include_attachment_text=False,  # Process attachments separately for chunking
                    max_total_attachment_text=None,  # Use config defaults
                )
                if conv is None:
                    logger.warning("Failed to load conversation %s, skipping", base_id)
                    continue
                meta = extract_metadata_lightweight(conv.get("manifest") or {})

                # Detect changed attachments (>= last run or stat() fails)
                changed_attachment_paths: set[Path] = set()
                last_ts = last_run_time.timestamp()
                for att_file in _iter_attachment_files(convo_dir):
                    try:
                        if att_file.stat().st_mtime >= last_ts:
                            changed_attachment_paths.add(att_file.resolve())
                    except OSError:
                        # Consider unreadable stat as changed to be safe
                        changed_attachment_paths.add(att_file.resolve())

                # Build mapping path->stable att_id for changed ones
                path_to_attid: dict[Path, str] = {}
                for att in conv.get("attachments") or []:
                    try:
                        path_to_attid[Path(att.get("path", "")).resolve()] = _att_id(
                            base_id, str(att.get("path", ""))
                        )
                    except Exception:
                        continue

                # Compute prefixes that changed due to mtime
                changed_prefixes: set[str] = set()
                for p in changed_attachment_paths:
                    aid = path_to_attid.get(p)
                    if aid:
                        changed_prefixes.add(aid)

                # Existing (current) attachment prefixes
                current_att_prefixes: set[str] = {
                    _att_id(base_id, str(att.get("path", "")))
                    for att in (conv.get("attachments") or [])
                }
                # Previously existing attachment prefixes for this conversation
                old_att_prefixes: set[str] = {
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
                    if (
                        prefix.startswith(f"{base_id}::att")
                        and prefix not in current_att_prefixes
                    ):
                        continue
                    # drop if attachment changed
                    if any(
                        did == pref or did.startswith(pref + "::")
                        for pref in changed_prefixes
                    ):
                        continue
                    unchanged_docs.append(d)

                # Rebuild changed attachments (respect per-conversation limit)
                added_for_conv = 0
                for att in conv.get("attachments") or []:
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
                                "attachment_size": (
                                    ap.stat().st_size if ap.exists() else None
                                ),
                                "subject": att_meta.get("subject") or "",
                                "from": att_meta.get("from", []),
                                "to": att_meta.get("to", []),
                                "cc": att_meta.get("cc", []),
                                "start_date": att_meta.get("start_date"),
                                "end_date": att_meta.get("end_date"),
                            }
                        )
                        ch["modified_time"] = mt
                        new_or_updated_docs.append(ch)
                        added_for_conv += 1
                # Skip the full rebuild below - conversation was handled via fast-path
                continue

        # If conversation changed or no last_run_time -> rebuild fully
        # HIGH #38: Use consistent load_conversation parameters
        conv = load_conversation(
            convo_dir,
            include_attachment_text=False,  # Process attachments separately for chunking
            max_total_attachment_text=None,  # Use config defaults
        )
        if conv is None:
            logger.warning("Failed to load conversation %s, skipping", base_id)
            continue
        meta = extract_metadata_lightweight(conv.get("manifest") or {})
        docs = _build_doc_entries(convo_dir, base_id, limit=limit, metadata=meta)
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
    existing_file_times: dict[str, float],
    existing_mapping: list[dict[str, Any]],
    limit: int | None = None,
) -> tuple[list[dict[str, Any]], set[str]]:
    """
    True incremental build using stored per-doc file times.
    Returns (new_or_updated_docs, deleted_doc_ids).

    Fixes:
      * Track deletions at the *chunk-level* by comparing existing mapping to current folder state.
      * Remove old chunks for changed attachments and conversations to avoid duplication.
      * Enforce per-conversation `limit` for newly rebuilt records.
    """
    new_docs: list[dict[str, Any]] = []
    deleted_ids: set[str] = set()

    max_mb, max_chars = _get_index_limits()

    # Group prior mapping rows by conversation and by 2-part prefix
    by_conv: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in existing_mapping or []:
        cid = str(row.get("conv_id") or "")
        if cid:
            by_conv[cid].append(row)

    by_conv_prefix: dict[str, dict[str, list[str]]] = {}
    for cid, rows in by_conv.items():
        pref_map: defaultdict[str, list[str]] = defaultdict(list)
        for r in rows:
            rid = str(r.get("id") or "")
            pref_map[_prefix_from_id(rid)].append(rid)
        by_conv_prefix[cid] = dict(pref_map)

    # Current conversations present on disk
    current_dirs = find_conversation_dirs(root)
    current_conv_ids: set[str] = {p.name for p in current_dirs}

    for convo_dir in current_dirs:
        base_id = convo_dir.name
        conv = load_conversation(convo_dir)
        if conv is None:
            logger.warning("Failed to load conversation %s, skipping", base_id)
            continue
        manifest = conv.get("manifest") or {}
        meta = extract_metadata_lightweight(manifest)

        # --- Conversation change detection ---
        conv_doc_id = f"{base_id}::conversation"
        conv_path = convo_dir / CONVERSATION_FILENAME
        try:
            conv_mtime = conv_path.stat().st_mtime
        except Exception:
            try:
                conv_mtime = convo_dir.stat().st_mtime
            except Exception as e:
                logger.debug("Failed to get file mtime, using current time: %s", e)
                conv_mtime = time.time()

        prior_times = [
            t
            for did, t in existing_file_times.items()
            if did == conv_doc_id or did.startswith(conv_doc_id + "::")
        ]
        prior_time = max(prior_times) if prior_times else None
        added_for_conv = 0

        # Enforce size cap for changed conversations: if oversized, drop old chunks and skip re-add
        conv_oversized = False
        try:
            if conv_path.exists():
                size_mb = conv_path.stat().st_size / (1024 * 1024)
                if size_mb > max_mb:
                    conv_oversized = True
        except Exception:
            conv_oversized = False

        if (prior_time or 0) < conv_mtime:
            # Delete old conv chunks first (to avoid duplication or stale content)
            for rid in by_conv_prefix.get(base_id, {}).get(conv_doc_id, []) or []:
                deleted_ids.add(rid)

            if conv_oversized:
                # Skip rebuilding this conversation due to size cap
                continue

            # Rebuild conversation
            txt = clean_email_text(conv.get("conversation_txt", ""))
            if max_chars and len(txt) > max_chars:
                txt = txt[:max_chars]
            chunks = prepare_index_units(
                txt,
                doc_id=conv_doc_id,
                doc_path=str(convo_dir / CONVERSATION_FILENAME),
                subject=meta.get("subject") or "",
                date=meta.get("end_date") or meta.get("start_date"),
            )
            for ch in chunks:
                if limit is not None and added_for_conv >= limit:
                    break
                ch.update(
                    {
                        "conv_id": base_id,
                        "doc_type": "conversation",
                        "subject": meta.get("subject") or "",
                        "from": meta.get("from", []),
                        "to": meta.get("to", []),
                        "cc": meta.get("cc", []),
                        "start_date": meta.get("start_date"),
                        "end_date": meta.get("end_date"),
                        "modified_time": conv_mtime,
                    }
                )
                new_docs.append(ch)
                added_for_conv += 1

        # --- Attachments change / deletion detection ---
        # Map current attachments (with stable IDs)
        current_att_prefixes: set[str] = set()
        for att in conv.get("attachments") or []:
            ap = Path(att.get("path", ""))
            prefix = _att_id(base_id, str(ap))
            current_att_prefixes.add(prefix)
            att_size_mb = None
            try:
                mt = ap.stat().st_mtime
                att_size_mb = ap.stat().st_size / (1024 * 1024)
            except Exception:
                mt = time.time()

            prior_times = [
                t
                for did, t in existing_file_times.items()
                if did == prefix or did.startswith(prefix + "::")
            ]
            prior_time = max(prior_times) if prior_times else None

            if (prior_time or 0) < mt and (att.get("text") or "").strip():
                # Always delete old chunks first
                for rid in by_conv_prefix.get(base_id, {}).get(prefix, []) or []:
                    deleted_ids.add(rid)

                # Enforce size cap: if oversized now, skip re-adding
                if att_size_mb is not None and att_size_mb > max_mb:
                    continue

                text = clean_email_text(att["text"])
                if max_chars and len(text) > max_chars:
                    text = text[:max_chars]
                chunks = prepare_index_units(
                    text,
                    doc_id=prefix,
                    doc_path=str(ap),
                    subject=meta.get("subject") or ap.name,
                    date=meta.get("end_date") or meta.get("start_date"),
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
                            "from": meta.get("from", []),
                            "to": meta.get("to", []),
                            "cc": meta.get("cc", []),
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
            p
            for p in prior_prefixes
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
def load_existing_index(
    index_dir: Path,
) -> tuple[
    Any | None, list[dict[str, Any]] | None, dict[str, float] | None, np.ndarray | None
]:
    """
    Returns (faiss_index_or_None, mapping_list_or_None, file_times_dict_or_None, embeddings_array_or_None)
    All parts are optional
    function handles missing files gracefully.
    """
    ixp = index_paths(index_dir)

    mapping = read_mapping(index_dir)

    file_times: dict[str, float] | None = None
    if ixp.file_times.exists():
        try:
            file_times = json.loads(ixp.file_times.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning("Failed to read %s: %s", ixp.file_times.name, e)

    embeddings: np.ndarray | None = None
    if ixp.embeddings.exists():
        try:
            # WINDOWS FIX: copy=True to close file handle immediately
            embeddings = np.load(str(ixp.embeddings), mmap_mode="r").astype(
                "float32", copy=True
            )
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
    """
    P0-5 FIX: Fallback consistency check with proper resource management.

    Uses safe_load_array to ensure memmap cleanup. Prevents Windows file locks.
    """
    try:
        ixp = index_paths(index_dir)
        m = read_mapping(index_dir) or []

        # Import safe_load_array from indexing_metadata
        try:
            from .indexing_metadata import safe_load_array
        except ImportError:
            from indexing_metadata import safe_load_array  # type: ignore

        # P0-5 FIX: Use context manager for embeddings
        if ixp.embeddings.exists():
            try:
                with safe_load_array(ixp.embeddings, mmap_mode="r") as emb:
                    if emb.shape[0] != len(m):
                        raise RuntimeError(
                            f"Embeddings/document count mismatch: {emb.shape[0]} vs {len(m)}"
                        ) from None

                    # FAISS ntotal vs embeddings
                    if HAVE_FAISS and faiss is not None and ixp.faiss.exists():
                        try:
                            fidx = faiss.read_index(str(ixp.faiss))  # type: ignore
                            if int(getattr(fidx, "ntotal", 0)) != int(emb.shape[0]):
                                raise RuntimeError(
                                    f"FAISS ntotal {getattr(fidx, 'ntotal', 0)} != embeddings {emb.shape[0]}"
                                ) from None
                        except Exception as e:
                            raise RuntimeError(f"FAISS consistency check failed: {e}") from None
            except FileNotFoundError:
                # Embeddings file removed between check and load
                raise RuntimeError("Embeddings file not found during consistency check") from None
    except Exception:
        raise


def save_index(
    index_dir: Path,
    embeddings: np.ndarray,
    mapping: list[dict[str, Any]],
    *,
    provider: str,
    num_folders: int,
) -> None:
    """
    P0-4 FIX: Persist index with transactional guarantees (atomic multi-file writes).

    All writes are atomic via IndexTransaction. Crash during write → automatic rollback.
    No partial state visible to readers. Post-save consistency check verifies alignment.

    Raises:
        FileOperationError: If transaction fails (includes automatic rollback)
        RuntimeError: If consistency check fails after successful write
    """
    ixp = index_paths(index_dir)
    ensure_dir(ixp.base)

    # P0-4 FIX: Check for crashed previous transaction and recover
    try:
        if (index_dir / ".wal").exists():
            logger.warning("Detected incomplete transaction, attempting recovery...")
            recover_index_from_wal(index_dir)
    except Exception as e:
        logger.error("WAL recovery failed: %s - proceeding with new transaction", e)

    # P0-4 FIX: Use transactional writes - all succeed or all rollback
    try:
        with IndexTransaction(index_dir) as txn:
            # 1) embeddings - staged in transaction
            txn.write_embeddings(embeddings, "embeddings.npy")

            # 2) mapping - staged in transaction
            txn.write_mapping(mapping, "mapping.json")

            # 3) metadata - staged in transaction
            meta = create_index_metadata(
                provider=provider,
                num_documents=len(mapping),
                num_folders=int(num_folders),
                index_dir=index_dir,
                custom_metadata={"actual_dimensions": int(embeddings.shape[1])},
            )
            txn.write_metadata(meta, "meta.json")

            # Auto-commit on context manager exit (if no exception)
            # Auto-rollback on any exception

        logger.info(
            "Transactionally saved index (vectors=%d, dimensions=%d)",
            embeddings.shape[0],
            embeddings.shape[1],
        )

    except Exception as e:
        logger.error("Index transaction failed and was rolled back: %s", e)
        raise

    # 4) Optional FAISS (written separately - not critical for correctness)
    # FAISS is a search optimization, not required for system correctness
    # If FAISS write fails, we log and continue (index is already committed)
    if HAVE_FAISS and faiss is not None:
        try:
            dim = int(embeddings.shape[1])
            index = faiss.IndexFlatIP(dim)  # type: ignore
            index.add(np.ascontiguousarray(embeddings, dtype=np.float32))  # type: ignore
            # Write FAISS to a temp path then replace
            faiss_tmp = ixp.faiss.with_suffix(ixp.faiss.suffix + ".tmp")
            faiss.write_index(index, str(faiss_tmp))  # type: ignore
            faiss_tmp.replace(ixp.faiss)
            logger.debug("FAISS index created successfully")
        except Exception as e:
            logger.warning("FAISS indexing failed, continuing without FAISS optimization: %s", e)

    # 5) Post-save consistency check (validates committed state)
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
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    ap = argparse.ArgumentParser(description="Build or update the email search index.")
    ap.add_argument(
        "--root",
        required=True,
        help="Export root that contains conversation folders and (optionally) an _index/ directory",
    )
    # Constrain provider to 'vertex' for this build to match metadata/validator support.
    ap.add_argument(
        "--provider",
        choices=["vertex"],
        default=os.getenv("EMBED_PROVIDER", "vertex"),
        help="Embedding provider for index build (this build supports only 'vertex')",
    )
    ap.add_argument(
        "--model", help="Embedding model/deployment override for the chosen provider"
    )
    ap.add_argument(
        "--batch",
        type=int,
        default=int(os.getenv("EMBED_BATCH", "64")),
        help="Embedding batch size (e.g., 64-250 for Vertex, will be clamped to max 250)",
    )
    ap.add_argument(
        "--index-root",
        help="Directory where the _index folder should be created (defaults to --root)",
    )
    ap.add_argument(
        "--force-reindex",
        action="store_true",
        help="Force a full re-index of all conversations",
    )
    ap.add_argument(
        "--limit",
        type=int,
        help="Limit number of chunks per conversation (for quick smoke tests)",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers for indexing (default: 1 for serial, use 6 for parallel with multiple GCP accounts)",
    )
    args = ap.parse_args()

    # Initialize GCP credentials for Vertex AI via config (single source of truth).
    if args.provider == "vertex":
        _initialize_gcp_credentials()

    root = Path(args.root).expanduser().resolve()
    index_base = (
        Path(args.index_root).expanduser().resolve() if args.index_root else root
    )

    if not index_base.exists():
        logger.info("Index base '%s' does not exist yet, creating it", index_base)
        index_base.mkdir(parents=True, exist_ok=True)

    out_dir = index_base / INDEX_DIRNAME_DEFAULT

    ensure_dir(out_dir)

    # Optional: pin model via env
    _apply_model_override(args.provider, args.model)

    # Load previous index bits (if any)
    _, existing_mapping, existing_file_times, existing_embeddings = load_existing_index(
        out_dir
    )
    existing_meta = load_index_metadata(out_dir)  # may be None

    # Last-run strategy
    last_run_time = None if args.force_reindex else _get_last_run_time(out_dir)

    # Provider resolution (prefer index provider to avoid dim mismatches)
    embed_provider = args.provider
    if not args.force_reindex and existing_embeddings is not None and existing_mapping:
        idx_provider = (
            (existing_meta.get("provider") or args.provider)
            if existing_meta
            else args.provider
        )
        if idx_provider and idx_provider != args.provider:
            logger.warning(
                "Using index provider '%s' instead of requested '%s' to ensure dimensional compatibility",
                idx_provider,
                args.provider,
            )
        embed_provider = idx_provider

    # ============================================================================
    # PARALLEL INDEXING PATH (when workers > 1)
    # ============================================================================
    if args.workers > 1:
        logger.info("Using parallel indexing with %d workers", args.workers)
        try:
            from .indexing_parallel import parallel_index_conversations

            merged_embeddings, merged_mapping = parallel_index_conversations(
                root=root,
                num_workers=args.workers,
                provider=embed_provider,
                limit=args.limit,
            )

            # Save merged results
            num_folders = len(
                {d["id"].split("::")[0] for d in merged_mapping if d.get("id")}
            )
            save_index(
                out_dir,
                merged_embeddings,
                merged_mapping,
                provider=embed_provider,
                num_folders=num_folders,
            )
            _save_run_time(out_dir)
            logger.info("Parallel index completed at %s", out_dir)
            return

        except ImportError as e:
            logger.warning(
                "Parallel indexing not available (%s), falling back to serial", e
            )
        except Exception as e:
            logger.error("Parallel indexing failed (%s), falling back to serial", e)

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
        )
        new_ids = {d["id"] for d in new_docs}
        unchanged_docs = [
            d
            for d in (existing_mapping or [])
            if d.get("id") not in new_ids and d.get("id") not in deleted_ids
        ]
        logger.info(
            "Incremental corpus: %d new/updated, %d unchanged, %d deleted",
            len(new_docs),
            len(unchanged_docs),
            len(deleted_ids),
        )
    else:
        # timestamp/manifest based
        if last_run_time:
            logger.info(
                "Starting incremental (timestamp) update from %s",
                last_run_time.strftime("%Y-%m-%d %H:%M:%S %Z"),
            )
        new_docs, unchanged_docs = build_corpus(
            root, out_dir, last_run_time=last_run_time, limit=args.limit
        )
        logger.info(
            "Corpus: %d new/updated, %d unchanged", len(new_docs), len(unchanged_docs)
        )

    # Short-circuit: nothing to do
    if (
        not new_docs
        and existing_embeddings is not None
        and existing_mapping
        and unchanged_docs
        and not args.force_reindex
    ):
        logger.info("No new content, index remains unchanged.")
        _save_run_time(out_dir)
        return

    # ------------------------------------------------------------------
    # 2) Embed texts (reuse unchanged vectors when possible)
    # ------------------------------------------------------------------
    final_docs: list[dict[str, Any]] = []
    all_embeddings: list[np.ndarray] = []

    if not (new_docs or unchanged_docs):
        logger.info("No documents to index.")
        _save_run_time(out_dir)
        return

    # Clamp batch size safely
    batch = max(1, min(int(args.batch or 64), EMBED_MAX_BATCH))

    def _validate_batch(vecs: np.ndarray, expected_rows: int) -> None:
        if vecs.size == 0:
            raise RuntimeError(
                "Embedding provider returned empty vectors, check provider credentials and model."
            )
        if vecs.ndim != 2 or vecs.shape[0] != int(expected_rows):
            raise RuntimeError(
                f"Invalid embeddings shape: got {vecs.shape}, expected rows={expected_rows}"
            ) from None
        if not np.isfinite(vecs).all():
            raise RuntimeError(
                "Invalid embeddings returned (non-finite values detected)"
            ) from None
        # At least one row must have a reasonable norm (~1.0 for unit-normalized embeddings)
        if float(np.max(np.linalg.norm(vecs, axis=1))) < 1e-3:
            raise RuntimeError("Embeddings look degenerate (all ~zero)") from None

    if existing_embeddings is not None and existing_mapping and not args.force_reindex:
        # Reuse unchanged vectors by id -> row index
        id_to_old_idx = {doc["id"]: i for i, doc in enumerate(existing_mapping)}
        unchanged_with_vecs: list[dict[str, Any]] = []
        unchanged_to_embed: list[dict[str, Any]] = []

        for d in unchanged_docs:
            idx = id_to_old_idx.get(d.get("id"))
            if idx is not None and 0 <= idx < existing_embeddings.shape[0]:
                unchanged_with_vecs.append(d)
                all_embeddings.append(existing_embeddings[idx : idx + 1])  # view slice
            else:
                unchanged_to_embed.append(d)

        # Prepare texts for embedding (fill from snippet or file path if needed)
        docs_to_embed = new_docs + unchanged_to_embed
        file_service = FileService(export_root=str(root))
        docs_to_embed = _materialize_text_for_docs(docs_to_embed, file_service)
        valid_docs = [d for d in docs_to_embed if str(d.get("text", "")).strip()]
        texts = [str(d["text"]) for d in valid_docs]

        if valid_docs:
            for i in range(0, len(texts), batch):
                chunk = texts[i : i + batch]
                vecs = embed_texts(chunk, provider=embed_provider)  # normalized vectors
                vecs = np.asarray(vecs, dtype="float32")
                _validate_batch(vecs, expected_rows=len(chunk))
                all_embeddings.append(vecs)

        final_docs = unchanged_with_vecs + valid_docs
    else:
        # Full embedding (fresh index or no reusable vectors)
        docs = list(new_docs + unchanged_docs)
        file_service = FileService(export_root=str(root))
        docs = _materialize_text_for_docs(docs, file_service)
        docs = [d for d in docs if str(d.get("text", "")).strip()]
        if not docs:
            logger.info("No non-empty documents to embed.")
            _save_run_time(out_dir)
            return

        texts = [str(d["text"]) for d in docs]
        for i in range(0, len(texts), batch):
            chunk = texts[i : i + batch]
            vecs = embed_texts(chunk, provider=embed_provider)
            vecs = np.asarray(vecs, dtype="float32")
            _validate_batch(vecs, expected_rows=len(chunk))
            all_embeddings.append(vecs)
        final_docs = docs

    if not all_embeddings:
        logger.info("Nothing to embed after filtering, aborting update.")
        _save_run_time(out_dir)
        return

    embeddings = np.vstack(all_embeddings)
    if embeddings.shape[0] != len(final_docs):
        raise RuntimeError(
            f"Embeddings/document count mismatch: {embeddings.shape[0]} vectors for {len(final_docs)} docs"
        ) from None

    # ------------------------------------------------------------------
    # 3) Persist index artifacts
    # ------------------------------------------------------------------
    mapping_out: list[dict[str, Any]] = []
    file_times: dict[str, float] = {}
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
                content_hash = hashlib.sha256(
                    text.encode("utf-8"), usedforsecurity=False
                ).hexdigest()[:16]  # First 64 bits
            except Exception:
                content_hash = ""

        # Extract emails from tuple format for mapping.json
        from_tuples = d.get("from", [])
        to_tuples = d.get("to", [])
        cc_tuples = d.get("cc", [])

        # Convert [(name, email), ...] to separate fields
        from_email = (
            from_tuples[0][1] if from_tuples and len(from_tuples[0]) > 1 else ""
        )
        from_name = from_tuples[0][0] if from_tuples and len(from_tuples[0]) > 0 else ""
        to_emails = [t[1] for t in to_tuples if len(t) > 1 and t[1]]
        cc_emails = [t[1] for t in cc_tuples if len(t) > 1 and t[1]]

        rec = {
            "id": d.get("id"),
            "path": d.get("path"),
            "conv_id": d.get("conv_id"),
            "doc_type": d.get("doc_type"),
            "subject": d.get("subject", ""),
            "date": d.get("date") or d.get("end_date") or d.get("start_date"),
            "start_date": d.get("start_date"),
            "end_date": d.get("end_date"),
            "from_email": from_email,
            "from_name": from_name,
            "to_emails": to_emails,
            "cc_emails": cc_emails,
            "attachment_name": d.get("attachment_name"),
            "attachment_type": d.get("attachment_type"),
            "attachment_size": d.get("attachment_size"),
            "snippet": snippet,
            "content_hash": content_hash,
        }
        mapping_out.append(rec)

        # Preserve previous modified_time for unchanged docs when missing
        doc_id = str(d.get("id"))
        mt = d.get("modified_time", None)
        if mt is None:
            mt = prior_times.get(doc_id, None)
        file_times[doc_id] = float(mt if mt is not None else time.time())

    save_index(
        out_dir,
        embeddings,
        mapping_out,
        provider=embed_provider,
        num_folders=len({(d["id"] or "").split("::")[0] for d in final_docs}),
    )

    # file_times.json (atomic) + timestamp (atomic)
    ixp = index_paths(out_dir)
    try:
        file_service = FileService(export_root=str(out_dir.parent))
        file_service.save_text_file(
            json.dumps(file_times, ensure_ascii=False, indent=2),
            ixp.file_times,
        )
    except Exception as e:
        logger.warning("Failed to write %s: %s", ixp.file_times.name, e)

    _save_run_time(out_dir)
    logger.info("Index updated at %s", out_dir)


if __name__ == "__main__":
    main()
