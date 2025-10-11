#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np

try:
    from .index_metadata import (  # filenames + helpers (single source of truth)  :contentReference[oaicite:20]{index=20}
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
    from .llm_client import (
        embed_texts,  # shim over runtime (unit-normalized embeddings)  :contentReference[oaicite:19]{index=19}
    )
    from .text_chunker import (
        prepare_index_units,  # emits id="doc_id::chunk{N}" when chunking  :contentReference[oaicite:21]{index=21}
    )
    from .utils import (  # library-safe logger  :contentReference[oaicite:18]{index=18}
        ensure_dir,
        find_conversation_dirs,
        load_conversation,
        logger,
    )
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
    from llm_client import embed_texts  # type: ignore
    from text_chunker import prepare_index_units  # type: ignore
    from utils import ensure_dir, find_conversation_dirs, load_conversation, logger  # type: ignore

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

# File name constants
CONVERSATION_FILENAME = "Conversation.txt"
MANIFEST_FILENAME = "manifest.json"
SUMMARY_FILENAME = "summary.json"

# ============================================================================
# GCP Credential Initialization
# ============================================================================

def _initialize_gcp_credentials() -> Optional[str]:
    """
    Initialize GCP credentials by preferring environment variables and then
    scanning a 'secrets' directory for any service-account JSON file.
    Returns the path to the credentials file if successful, None otherwise.
    """
    # 1) Prefer explicit environment variable
    env_creds = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if env_creds:
        creds_path = Path(env_creds)
        if creds_path.exists():
            logger.info("Using existing GCP credentials from environment")
            return str(creds_path)

    # 2) Allow override for secrets directory, default to repo's ../secrets
    secrets_base = os.getenv("SECRETS_DIR")
    secrets_dir = Path(secrets_base).expanduser() if secrets_base else Path(__file__).parent.parent / "secrets"
    if not secrets_dir.exists():
        logger.warning("Secrets directory not found at %s", secrets_dir)
        return None

    # 3) Optional preference list via env (colon-separated basenames)
    preferred = (os.getenv("PREFERRED_GCP_CREDENTIALS") or "").split(":")
    preferred = [p for p in preferred if p]

    candidates = []
    for name in preferred:
        p = secrets_dir / name
        if p.exists():
            candidates.append(p)

    # If no preferred files, try any *.json in the secrets dir
    if not candidates:
        candidates = sorted(secrets_dir.glob("*.json"))

    for cred_path in candidates:
        try:
            with open(cred_path, "r") as f:
                creds_data = json.load(f)
            if "project_id" in creds_data and "client_email" in creds_data:
                project_id = str(creds_data.get("project_id"))
                # Set environment variables for GCP authentication
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(cred_path)
                os.environ["GCP_PROJECT"] = project_id
                os.environ["GOOGLE_CLOUD_PROJECT"] = project_id
                os.environ["VERTEX_PROJECT"] = project_id
                # Set default location if not already set
                os.environ.setdefault("GCP_REGION", "us-central1")
                os.environ.setdefault("VERTEX_LOCATION", "us-central1")
                logger.info("Initialized GCP credentials from secrets directory")
                return str(cred_path)
        except json.JSONDecodeError as e:
            logger.warning("Invalid JSON in credentials file '%s': %s", cred_path.name, e)
        except Exception as e:
            logger.warning("Error reading credentials file '%s': %s", cred_path.name, e)

    logger.error("No valid GCP credentials found in secrets directory")
    return None

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
    (index_dir / TIMESTAMP_FILENAME).write_text(datetime.now(timezone.utc).isoformat(), encoding="utf-8")


def _clean_index_text(text: str) -> str:
    """
    Light cleaning to reduce noise for embeddings, without altering meaning.
    (Aggressive email cleanup happens upstream when needed.)
    """
    if not text:
        return ""
    s = re.sub(r"[ \t]+", " ", text)
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


def _safe_read_text(path: Path, *, max_chars: int = 200_000) -> str:
    try:
        txt = path.read_text(encoding="utf-8", errors="ignore")
        return txt[:max_chars]
    except Exception:
        return ""


def _materialize_text_for_docs(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Ensure each doc has a non-empty 'text' field, using:
      1) existing 'text', else
      2) 'snippet' from prior mapping, else
      3) file content at 'path' (bounded)
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
                    d["text"] = _safe_read_text(p)
        out.append(d)
    return out


def _prefix_from_id(doc_id: str) -> str:
    """
    Normalize a doc id to its 2-part prefix:
      "<conv_id>::conversation" or "<conv_id>::attN"
    If no '::' present, returns the whole id.
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


# ============================================================================
# Chunk-building for a conversation (respects per-conversation limit)
# ============================================================================
def _build_doc_entries(
    conv: Dict[str, Any],
    convo_dir: Path,
    base_id: str,
    limit: Optional[int] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Build per-document (conversation + attachments) index records for a conversation folder.
    Enforces `limit` across conversation+attachments combined.
    """
    metadata = dict(metadata or {})
    out: List[Dict[str, Any]] = []

    # Conversation main text
    convo_txt = _clean_index_text(conv.get("conversation_txt", ""))
    if convo_txt:
        conv_id = f"{base_id}::conversation"
        conv_chunks = prepare_index_units(
            convo_txt,
            doc_id=conv_id,
            doc_path=str(convo_dir / CONVERSATION_FILENAME),
            subject=metadata.get("subject") or "",
            date=metadata.get("end_date") or metadata.get("start_date"),
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
                }
            )
            out.append(ch)

    # Attachments (respect limit across attachments and chunks)
    for i, att in enumerate(conv.get("attachments") or [], start=1):
        if limit is not None and len(out) >= limit:
            break
        ap = Path(att.get("path", ""))
        text = _clean_index_text(att.get("text", "") or "")
        if not text:
            continue
        att_id = f"{base_id}::att{i}"
        att_chunks = prepare_index_units(
            text,
            doc_id=att_id,
            doc_path=str(ap),
            subject=metadata.get("subject") or ap.name,
            date=metadata.get("end_date") or metadata.get("start_date"),
        )
        for ch in att_chunks:
            if limit is not None and len(out) >= limit:
                break
            ch.update(
                {
                    "conv_id": base_id,
                    "doc_type": "attachment",
                    "attachment_name": ap.name,
                    "attachment_type": ap.suffix.lstrip(".").lower(),
                    "attachment_size": os.path.getsize(ap) if ap.exists() else None,
                    "subject": metadata.get("subject") or "",
                    "participants": metadata.get("participants") or [],
                    "start_date": metadata.get("start_date"),
                    "end_date": metadata.get("end_date"),
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

                # Build mapping path->index (1-based) for changed ones
                path_to_index: Dict[Path, int] = {}
                for j, att in enumerate(conv.get("attachments") or [], start=1):
                    try:
                        path_to_index[Path(att.get("path", "")).resolve()] = j
                    except Exception:
                        continue

                # Compute prefixes that changed due to mtime
                changed_prefixes: Set[str] = set()
                for p in changed_attachment_paths:
                    idx = path_to_index.get(p)
                    if idx is not None:
                        changed_prefixes.add(f"{base_id}::att{idx}")

                # Existing (current) attachment prefixes
                current_att_prefixes: Set[str] = {
                    f"{base_id}::att{j}" for j in range(1, len(conv.get("attachments") or []) + 1)
                }
                # Previously existing attachment prefixes for this conversation
                old_att_prefixes: Set[str] = {
                    _prefix_from_id(d.get("id", ""))
                    for d in prior_docs_for_conv
                    if str(d.get("id", "")).startswith(f"{base_id}::att")
                }
                # Any newly-added attachments (present now but not before) must be rebuilt
                for pref in current_att_prefixes - old_att_prefixes:
                    try:
                        idx = int(pref.split("::att")[1])
                    except Exception:
                        idx = None
                    if idx is not None:
                        try:
                            ap = Path((conv.get("attachments") or [])[idx - 1].get("path", "")).resolve()
                            changed_attachment_paths.add(ap)
                            changed_prefixes.add(pref)
                        except Exception:
                            pass

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
                for p in changed_attachment_paths:
                    idx = path_to_index.get(p)
                    if idx is None:
                        continue
                    try:
                        att = (conv.get("attachments") or [])[idx - 1]
                    except Exception:
                        att = None
                    if not att or not (att.get("text") or "").strip():
                        continue
                    att_meta = dict(meta)
                    att_meta["subject"] = att_meta.get("subject") or Path(att.get("path", "")).name
                    chunks = prepare_index_units(
                        _clean_index_text(att.get("text", "")),
                        doc_id=f"{base_id}::att{idx}",
                        doc_path=str(att.get("path", "")),
                        subject=att_meta.get("subject") or "",
                        date=att_meta.get("end_date") or att_meta.get("start_date"),
                    )
                    for ch in chunks:
                        if limit is not None and added_for_conv >= limit:
                            break
                        ch.update(
                            {
                                "conv_id": base_id,
                                "doc_type": "attachment",
                                "attachment_name": Path(att.get("path", "")).name,
                                "attachment_type": Path(att.get("path", "")).suffix.lstrip(".").lower(),
                                "attachment_size": Path(att.get("path", "")).stat().st_size
                                if Path(att.get("path", "")).exists()
                                else None,
                                "subject": att_meta.get("subject") or "",
                                "participants": att_meta.get("participants") or [],
                                "start_date": att_meta.get("start_date"),
                                "end_date": att_meta.get("end_date"),
                            }
                        )
                        try:
                            ch["modified_time"] = Path(att.get("path", "")).stat().st_mtime
                        except Exception:
                            ch["modified_time"] = time.time()
                        new_or_updated_docs.append(ch)
                        added_for_conv += 1
                continue  # next conversation

        # If conversation changed or no last_run_time -> rebuild fully
        conv = load_conversation(convo_dir)
        meta = _extract_manifest_metadata(conv)
        docs = _build_doc_entries(conv, convo_dir, base_id, limit=limit, metadata=meta)
        try:
            mt = convo_txt_path.stat().st_mtime
        except Exception:
            mt = time.time()
        for d in docs:
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
    limit: Optional[int] = None
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
            txt = _clean_index_text(conv.get("conversation_txt", ""))
            chunks = prepare_index_units(
                txt,
                doc_id=conv_doc_id,
                doc_path=str(convo_dir / CONVERSATION_FILENAME),
                subject=meta.get("subject") or "",
                date=meta.get("end_date") or meta.get("start_date"),
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
        # Map current attachments
        current_att_prefixes: Set[str] = set()
        for i, att in enumerate(conv.get("attachments") or [], start=1):
            prefix = f"{base_id}::att{i}"
            current_att_prefixes.add(prefix)
            ap = Path(att.get("path", ""))
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

                text = _clean_index_text(att["text"])
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
                    ch.update(
                        {
                            "conv_id": base_id,
                            "doc_type": "attachment",
                            "attachment_name": ap.name,
                            "attachment_type": ap.suffix.lstrip(".").lower(),
                            "attachment_size": ap.stat().st_size if ap.exists() else None,
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


def save_index(index_dir: Path, embeddings: np.ndarray, mapping: List[Dict[str, Any]], *, provider: str, num_folders: int) -> None:
    """
    Persist embeddings (embeddings.npy), mapping.json, (optional) FAISS index, and meta.json.
    """
    ixp = index_paths(index_dir)
    ensure_dir(ixp.base)

    # 1) embeddings
    np.save(str(ixp.embeddings), embeddings.astype("float32", order="C"))

    # 2) mapping
    write_mapping(index_dir, mapping)

    # 3) optional FAISS (Inner Product index – cosine ready because vectors are unit-normalized)
    if HAVE_FAISS and faiss is not None:
        try:
            dim = int(embeddings.shape[1])
            index = faiss.IndexFlatIP(dim)  # type: ignore
            index.add(np.ascontiguousarray(embeddings, dtype=np.float32))  # type: ignore
            faiss.write_index(index, str(ixp.faiss))  # type: ignore
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


# ============================================================================
# CLI
# ============================================================================
def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    ap = argparse.ArgumentParser(description="Build or update the email search index.")
    ap.add_argument("--root", required=True, help="Export root that contains conversation folders and (optionally) an _index/ directory")
    ap.add_argument("--provider", choices=["vertex", "openai", "azure", "cohere", "huggingface", "local", "qwen"], default=os.getenv("EMBED_PROVIDER", "vertex"), help="Embedding provider for index build")
    ap.add_argument("--model", help="Embedding model/deployment override for the chosen provider")
    ap.add_argument("--batch", type=int, default=int(os.getenv("EMBED_BATCH", "64")), help="Embedding batch size (e.g., 64–250 for Vertex, will be clamped to max 250)")
    ap.add_argument("--index-root", help="Directory where the _index folder should be created (defaults to --root)")
    ap.add_argument("--force-reindex", action="store_true", help="Force a full re-index of all conversations")
    ap.add_argument("--limit", type=int, help="Limit number of chunks per conversation (for quick smoke tests)")
    args = ap.parse_args()
    
    # Initialize GCP credentials if using Vertex AI
    if args.provider == "vertex":
        creds_path = _initialize_gcp_credentials()
        if not creds_path:
            logger.error("Failed to initialize GCP credentials for Vertex AI")
            logger.info("Please ensure valid service account JSON files exist in the 'secrets' directory")
            logger.info("Expected files: embed2-474114-fca38b4d2068.json or other service account JSON files")
            return

    root = Path(args.root).expanduser().resolve()
    index_base = Path(args.index_root).expanduser().resolve() if args.index_root else root

    if not index_base.exists():
        logger.info("Index base '%s' does not exist yet; creating it", index_base)
        index_base.mkdir(parents=True, exist_ok=True)

    out_dir = index_base / INDEX_DIRNAME_DEFAULT

    ensure_dir(out_dir)

    # Optional: pin model via env
    _apply_model_override(args.provider, args.model)

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
            limit=args.limit
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
        new_docs, unchanged_docs = build_corpus(root, out_dir, last_run_time=last_run_time, limit=args.limit)
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
                if vecs.size == 0:
                    raise RuntimeError("Embedding provider returned empty vectors; check provider credentials and model.")
                all_embeddings.append(np.asarray(vecs, dtype="float32"))

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
            if vecs.size == 0:
                raise RuntimeError("Embedding provider returned empty vectors; check provider credentials and model.")
            all_embeddings.append(np.asarray(vecs, dtype="float32"))
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
            "to_emails": d.get("to_emails", []),  # SCHEMA FIX: Changed from to_recipients to to_emails
            "cc_emails": d.get("cc_emails", []),  # SCHEMA FIX: Changed from cc_recipients to cc_emails
            "participants": d.get("participants", []),  # SCHEMA FIX: Added missing participants field
            "attachment_name": d.get("attachment_name"),
            "attachment_type": d.get("attachment_type"),
            "attachment_size": d.get("attachment_size"),
            "snippet": snippet,
        }
        mapping_out.append(rec)

        # Preserve previous modified_time for unchanged docs when missing
        doc_id = str(d.get("id"))
        mt = d.get("modified_time", None)
        if mt is None:
            mt = prior_times.get(doc_id, None)
        file_times[doc_id] = float(mt if mt is not None else time.time())

    save_index(out_dir, embeddings, mapping_out, provider=embed_provider, num_folders=len({(d["id"] or "").split("::")[0] for d in final_docs}))

    # file_times.json + timestamp
    ixp = index_paths(out_dir)
    try:
        ixp.file_times.write_text(json.dumps(file_times, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        logger.warning("Failed to write %s: %s", ixp.file_times.name, e)

    _save_run_time(out_dir)
    logger.info("Index updated at %s", out_dir)


if __name__ == "__main__":
    main()
