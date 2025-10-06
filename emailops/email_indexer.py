#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np

from .utils import ensure_dir, find_conversation_dirs, load_conversation, logger  # fileciteturn0file6
from .llm_client import embed_texts  # fileciteturn0file1
from .index_metadata import (  # fileciteturn0file3
    FILE_TIMES_FILENAME,
    FAISS_INDEX_FILENAME,
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
from .text_chunker import prepare_index_units  # fileciteturn0file4

try:
    import faiss  # type: ignore
    HAVE_FAISS = True
except Exception:
    faiss = None  # type: ignore
    HAVE_FAISS = False


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
    Apply very light cleaning to reduce noise for embeddings, without altering meaning.
    Do NOT remove headers wholesale here—utils.clean_email_text is applied upstream when needed.
    """
    if not text:
        return ""
    # collapse whitespace runs and common boilerplate
    s = re.sub(r"[ \t]+", " ", text)
    s = re.sub(r"\n{3,}", "\n\n", s)
    # trim long dashes/separators
    s = re.sub(r"[=\-_]{10,}", " ", s)
    # drop very long repeated header ruler lines
    s = re.sub(r"(?m)^-{20,}\n", "", s)
    return s.strip()


def _extract_manifest_metadata(conv: Dict[str, Any]) -> Dict[str, Any]:
    """Pick a handful of safe fields from manifest for indexing metadata."""
    man = conv.get("manifest") or {}
    subject = (man.get("smart_subject") or man.get("subject") or "").strip()
    # Participants (strings only)
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


def _build_doc_entries(
    conv: Dict[str, Any],
    convo_dir: Path,
    base_id: str,
    limit: Optional[int] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Build per-document (conversation + attachments) index records for a conversation folder.
    - Conversation is chunked when large (via text_chunker.prepare_index_units).
    - Attachments are chunked similarly (only textful ones).
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
            doc_path=str(convo_dir / "Conversation.txt"),
            subject=metadata.get("subject") or "",
            date=metadata.get("end_date") or metadata.get("start_date"),
        )
        for ch in conv_chunks[: (limit or len(conv_chunks))]:
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

    # Attachments
    for i, att in enumerate(conv.get("attachments") or [], start=1):
        if limit and len(out) >= limit:
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


def _iter_attachment_files(convo_dir: Path) -> Iterable[Path]:
    attachments_dir = convo_dir / "Attachments"
    if attachments_dir.exists():
        for p in attachments_dir.rglob("*"):
            if p.is_file():
                yield p
    # Also treat any file directly in convo_dir (except known files) as potential attachment
    for child in convo_dir.iterdir():
        if child.is_file() and child.name not in {"Conversation.txt", "manifest.json", "summary.json"}:
            yield child


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
    old_mapping: Dict[str, Dict[str, Any]] = {d.get("id"): d for d in (old_mapping_list or [])}

    for convo_dir in conversation_dirs:
        base_id = convo_dir.name
        # If Conversation.txt missing (rare/corrupt), skip the folder entirely
        convo_txt_path = convo_dir / "Conversation.txt"
        if not convo_txt_path.exists():
            continue

        needs_processing = True
        if last_run_time is not None:
            try:
                modified_time = datetime.fromtimestamp(convo_txt_path.stat().st_mtime, tz=timezone.utc)
            except Exception:
                modified_time = None

            if modified_time and modified_time < last_run_time:
                # Conversation main text has not changed since last run — now check attachments.
                needs_processing = False

                # Inventory old docs for this conversation
                convo_prefix = f"{base_id}::"
                base_unchanged = [doc for did, doc in old_mapping.items() if isinstance(did, str) and did.startswith(convo_prefix)]

                # Detect changed attachments (>= last run or missing now)
                changed_attachment_paths: Set[Path] = set()
                last_ts = last_run_time.timestamp()
                for att_file in _iter_attachment_files(convo_dir):
                    try:
                        if att_file.stat().st_mtime >= last_ts:
                            changed_attachment_paths.add(att_file.resolve())
                    except OSError:
                        # if stat fails, treat as changed (could be transient delete)
                        changed_attachment_paths.add(att_file.resolve())

                # Load current manifest/attachments to (a) build new docs for changed attachments
                # and (b) filter out docs for attachments that no longer exist.
                conv = load_conversation(convo_dir)
                meta = _extract_manifest_metadata(conv)
                # Build mapping path->index (1-based) for changed ones
                path_to_index: Dict[Path, int] = {}
                for j, att in enumerate(conv.get("attachments") or [], start=1):
                    try:
                        path_to_index[Path(att.get("path", "")).resolve()] = j
                    except Exception:
                        continue

                # Compute prefixes for changed attachments (present in manifest)
                changed_prefixes: Set[str] = set()
                for p in changed_attachment_paths:
                    idx = path_to_index.get(p)
                    if idx is not None:
                        changed_prefixes.add(f"{base_id}::att{idx}")

                # Also compute prefixes for all attachments that still exist *now* (for removal of stale docs)
                existing_att_prefixes: Set[str] = {f"{base_id}::att{j}" for j in range(1, len(conv.get("attachments") or []) + 1)}

                # Keep only truly unchanged docs (not changed AND still present)
                for d in base_unchanged:
                    did = d.get("id", "")
                    # derive the 2-part prefix id::<attX|conversation>
                    parts = did.split("::")
                    prefix = "::".join(parts[:2]) if len(parts) >= 2 else did
                    # drop if attachment no longer present
                    if prefix.startswith(f"{base_id}::att") and prefix not in existing_att_prefixes:
                        continue
                    # drop if attachment changed
                    if any(did == pref or did.startswith(pref + "::") for pref in changed_prefixes):
                        continue
                    unchanged_docs.append(d)

                # For attachments that changed: rebuild
                for p in changed_attachment_paths:
                    idx = path_to_index.get(p)
                    if idx is None:
                        # Attachment removed or no longer present: do nothing (docs above were filtered out)
                        continue
                    # find its entry in conv.attachments (by 1-based index)
                    att = None
                    try:
                        att = (conv.get("attachments") or [])[idx - 1]
                    except Exception:
                        att = None
                    if att and (att.get("text") or "").strip():
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
                            # use file mtime for incremental write tracking
                            try:
                                ch["modified_time"] = Path(att.get("path", "")).stat().st_mtime
                            except Exception:
                                ch["modified_time"] = time.time()
                            new_or_updated_docs.append(ch)

                # If there were no changes at all (attachments unchanged and present), we already
                # copied all unchanged docs above and can continue.
                continue  # next conversation

        # If we reach here, either conversation changed or no last_run_time -> rebuild fully
        conv = load_conversation(convo_dir)
        meta = _extract_manifest_metadata(conv)
        docs = _build_doc_entries(conv, convo_dir, base_id, limit=limit, metadata=meta)
        # mark modified time for every built record to assist write-out of file_times
        try:
            mt = convo_txt_path.stat().st_mtime
        except Exception:
            mt = time.time()
        for d in docs:
            d["modified_time"] = mt
        new_or_updated_docs.extend(docs)

    return new_or_updated_docs, unchanged_docs


# ============================================================================
# File-times incremental (more precise; supports deletions)
# ============================================================================
def build_incremental_corpus(
    root: Path, existing_file_times: Dict[str, float], limit: Optional[int] = None
) -> Tuple[List[Dict[str, Any]], Set[str]]:
    """
    True incremental build using stored per-doc file times.
    Returns (new_or_updated_docs, deleted_doc_ids).
    """
    new_docs: List[Dict[str, Any]] = []
    current_doc_ids: Set[str] = set()

    for convo_dir in find_conversation_dirs(root):
        base_id = convo_dir.name
        conv = load_conversation(convo_dir)
        meta = _extract_manifest_metadata(conv)

        # Conversation doc id and change detection
        conv_doc_id = f"{base_id}::conversation"
        current_doc_ids.add(conv_doc_id)
        try:
            conv_mtime = (convo_dir / "Conversation.txt").stat().st_mtime
        except Exception:
            try:
                conv_mtime = convo_dir.stat().st_mtime
            except Exception:
                conv_mtime = time.time()

        existing_time = max(
            (t for did, t in existing_file_times.items() if did == conv_doc_id or did.startswith(conv_doc_id + "::")),
            default=None,
        )
        if (existing_time or 0) < conv_mtime:
            # rebuild conversation
            txt = _clean_index_text(conv.get("conversation_txt", ""))
            chunks = prepare_index_units(
                txt,
                doc_id=conv_doc_id,
                doc_path=str(convo_dir / "Conversation.txt"),
                subject=meta.get("subject") or "",
                date=meta.get("end_date") or meta.get("start_date"),
            )
            for ch in chunks:
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

        # Attachments
        for i, att in enumerate(conv.get("attachments") or [], start=1):
            aid = f"{base_id}::att{i}"
            current_doc_ids.add(aid)
            ap = Path(att.get("path", ""))
            try:
                mt = ap.stat().st_mtime
            except Exception:
                mt = time.time()
            existing_time = max(
                (t for did, t in existing_file_times.items() if did == aid or did.startswith(aid + "::")), default=None
            )
            if (existing_time or 0) < mt and (att.get("text") or "").strip():
                text = _clean_index_text(att["text"])
                chunks = prepare_index_units(
                    text,
                    doc_id=aid,
                    doc_path=str(ap),
                    subject=meta.get("subject") or ap.name,
                    date=meta.get("end_date") or meta.get("start_date"),
                )
                for ch in chunks:
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

    # Anything not present now but in existing_file_times is deleted
    deleted_ids = set(existing_file_times.keys()) - current_doc_ids
    return new_docs, deleted_ids


# ============================================================================
# Index I/O
# ============================================================================
def load_existing_index(index_dir: Path) -> Tuple[Optional[Any], Optional[List[Dict[str, Any]]], Optional[Dict[str, float]], Optional[np.ndarray]]:
    """
    Returns (faiss_index_or_None, mapping_list_or_None, file_times_dict_or_None, embeddings_array_or_None)
    All parts are optional; function handles missing files gracefully.
    """
    ixp = index_paths(index_dir)  # paths helper

    # mapping.json
    mapping = read_mapping(index_dir)

    # file_times.json
    file_times: Optional[Dict[str, float]] = None
    if ixp.file_times.exists():
        try:
            file_times = json.loads(ixp.file_times.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning("Failed to read %s: %s", ixp.file_times.name, e)

    # embeddings.npy (mmap to avoid memory blow-ups)
    embeddings: Optional[np.ndarray] = None
    if ixp.embeddings.exists():
        try:
            embeddings = np.load(str(ixp.embeddings), mmap_mode="r").astype("float32", copy=False)
        except Exception as e:
            logger.warning("Failed to read %s: %s", ixp.embeddings.name, e)

    # faiss index (optional)
    fidx = None
    if HAVE_FAISS and ixp.faiss.exists():
        try:
            fidx = faiss.read_index(str(ixp.faiss))
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
    if HAVE_FAISS:
        try:
            dim = int(embeddings.shape[1])
            index = faiss.IndexFlatIP(dim)
            index.add(np.ascontiguousarray(embeddings))
            faiss.write_index(index, str(ixp.faiss))
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
    ap.add_argument("--batch", type=int, default=int(os.getenv("EMBED_BATCH", "64")), help="Embedding batch size (e.g., 64–250 for Vertex)")
    ap.add_argument("--force-reindex", action="store_true", help="Force a full re-index of all conversations")
    ap.add_argument("--limit", type=int, help="Limit number of chunks per conversation (for quick smoke tests)")
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    out_dir = root / INDEX_DIRNAME_DEFAULT  # fixed: use constant from index_metadata

    ensure_dir(out_dir)

    # Optional: pin model via env
    _apply_model_override(args.provider, args.model)

    # Load previous index bits (if any)
    existing_faiss, existing_mapping, existing_file_times, existing_embeddings = load_existing_index(out_dir)
    existing_meta = load_index_metadata(out_dir)  # may be None

    # Figure out last-run strategy
    last_run_time = None if args.force_reindex else _get_last_run_time(out_dir)

    # Choose embedding provider to ensure dimensional consistency on incremental runs
    embed_provider = args.provider
    if not args.force_reindex and existing_embeddings is not None and existing_mapping:
        # Always use the provider recorded in the existing index to avoid dim mismatches
        idx_provider = (existing_meta.get("provider") or args.provider) if existing_meta else args.provider
        if idx_provider and idx_provider != args.provider:
            logger.warning("Existing index was built with provider '%s'; using it for incremental update to avoid dimension mismatches.", idx_provider)
        embed_provider = idx_provider

    # ------------------------------------------------------------------
    # 1) Build doc corpus (new/updated + unchanged)
    # ------------------------------------------------------------------
    if existing_file_times and not args.force_reindex:
        # precise incremental
        new_docs, deleted_ids = build_incremental_corpus(root, existing_file_times, limit=args.limit)
        new_ids = {d["id"] for d in new_docs}
        unchanged_docs = [d for d in (existing_mapping or []) if d.get("id") not in new_ids and d.get("id") not in deleted_ids]
        logger.info("Incremental corpus: %d new/updated, %d unchanged, %d deleted", len(new_docs), len(unchanged_docs), len(deleted_ids))
    else:
        # timestamp/manifest based (or full rebuild on first run)
        if last_run_time:
            logger.info("Starting incremental (timestamp) update from %s", last_run_time.strftime("%Y-%m-%d %H:%M:%S %Z"))
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

    if existing_embeddings is not None and existing_mapping and not args.force_reindex:
        # Reuse unchanged vectors by id -> row index
        id_to_old_idx = {doc["id"]: i for i, doc in enumerate(existing_mapping)}
        unchanged_with_vecs: List[Dict[str, Any]] = []
        unchanged_to_embed: List[Dict[str, Any]] = []

        for d in unchanged_docs:
            idx = id_to_old_idx.get(d.get("id"))
            if idx is not None and 0 <= idx < existing_embeddings.shape[0]:
                unchanged_with_vecs.append(d)
                all_embeddings.append(existing_embeddings[idx : idx + 1])  # view slice
            else:
                unchanged_to_embed.append(d)

        # Embed new + those unchanged that lacked vectors
        docs_to_embed = new_docs + unchanged_to_embed
        valid_docs = [d for d in docs_to_embed if str(d.get("text", "")).strip()]
        texts = [str(d["text"]) for d in valid_docs]

        if valid_docs:
            batch = max(1, int(args.batch or 64))
            for i in range(0, len(texts), batch):
                chunk = texts[i : i + batch]
                vecs = embed_texts(chunk, provider=embed_provider)  # normalized vectors
                if vecs.size == 0:
                    raise RuntimeError("Embedding provider returned empty vectors; check provider credentials and model.")
                all_embeddings.append(vecs.astype("float32", copy=False))

        final_docs = unchanged_with_vecs + valid_docs
    else:
        # Full embedding (fresh index or no reusable vectors)
        docs = [d for d in (new_docs + unchanged_docs) if str(d.get("text", "")).strip()]
        if len(docs) != (len(new_docs) + len(unchanged_docs)):
            logger.warning("Filtered out %d empty-text documents before embedding.", (len(new_docs) + len(unchanged_docs)) - len(docs))
        if not docs:
            logger.info("No non-empty documents to embed.")
            _save_run_time(out_dir)
            return

        batch = max(1, int(args.batch or 64))
        texts = [str(d["text"]) for d in docs]
        for i in range(0, len(texts), batch):
            chunk = texts[i : i + batch]
            vecs = embed_texts(chunk, provider=embed_provider)
            if vecs.size == 0:
                raise RuntimeError("Embedding provider returned empty vectors; check provider credentials and model.")
            all_embeddings.append(vecs.astype("float32", copy=False))
        final_docs = docs

    # Stack vectors and validate alignment
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
    # Build mapping.json entries (keep only required fields; add snippet for UX)
    mapping_out: List[Dict[str, Any]] = []
    file_times: Dict[str, float] = {}
    for d in final_docs:
        text = str(d.get("text", ""))
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
            "to_recipients": d.get("to_recipients", []),
            "cc_recipients": d.get("cc_recipients", []),
            "attachment_name": d.get("attachment_name"),
            "attachment_type": d.get("attachment_type"),
            "attachment_size": d.get("attachment_size"),
            "snippet": text[:500],
        }
        mapping_out.append(rec)
        file_times[str(d.get("id"))] = float(d.get("modified_time", time.time()))

    # Save everything
    save_index(out_dir, embeddings, mapping_out, provider=embed_provider, num_folders=len({(d["id"] or "").split("::")[0] for d in final_docs}))

    # file_times.json + timestamp (atomic enough for our use)
    ixp = index_paths(out_dir)
    try:
        ixp.file_times.write_text(json.dumps(file_times, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        logger.warning("Failed to write %s: %s", ixp.file_times.name, e)
    _save_run_time(out_dir)
    logger.info("Index updated at %s", out_dir)


if __name__ == "__main__":
    main()
