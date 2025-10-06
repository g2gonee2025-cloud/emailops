#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, os, logging, re, time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Iterable, Set
from datetime import datetime, timezone

import numpy as np

from .utils import logger, find_conversation_dirs, load_conversation, ensure_dir
from .llm_client import embed_texts
from .index_metadata import create_index_metadata, save_index_metadata
from .text_chunker import chunk_for_indexing, should_chunk_text

try:
    import faiss
    HAVE_FAISS = True
except ImportError:
    faiss = None
    HAVE_FAISS = False

INDEX_DIRNAME = os.getenv("INDEX_DIRNAME", "_index")
INDEX_NAME = "index.faiss"
MAPPING_NAME = "mapping.json"
TIMESTAMP_NAME = "last_run.txt"
FILE_TIMES_NAME = "file_times.json"

def _apply_model_override(provider: str, model: Optional[str]) -> None:
    """
    Map --model to the correct provider env var so llm_client picks it up.
    """
    if not model:
        return
    env_map = {
        "vertex": "VERTEX_EMBED_MODEL",
        "openai": "OPENAI_EMBED_MODEL",
        "azure": "AZURE_OPENAI_DEPLOYMENT",
        "cohere": "COHERE_EMBED_MODEL",
        "huggingface": "HF_EMBED_MODEL",
        "local": "LOCAL_EMBED_MODEL",
        "qwen": "QWEN_EMBED_MODEL",
    }
    env_key = env_map.get(provider)
    if env_key:
        os.environ[env_key] = model
        logger.info(f"Overriding {env_key}={model}")
    else:
        logger.warning(f"No env override mapping for provider '{provider}'")

def _get_last_run_time(index_dir: Path) -> datetime | None:
    """Reads the timestamp of the last successful run."""
    timestamp_file = index_dir / TIMESTAMP_NAME
    if not timestamp_file.exists():
        return None
    try:
        return datetime.fromisoformat(timestamp_file.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return None

def _save_run_time(index_dir: Path) -> None:
    """Saves the timestamp of the current run."""
    ensure_dir(index_dir)
    timestamp_file = index_dir / TIMESTAMP_NAME
    timestamp_file.write_text(datetime.now(timezone.utc).isoformat(), encoding="utf-8")

def _sanitize_text(text: str) -> str:
    """Applies cleaning rules to the raw email text."""
    # Remove the "Email Conversation Export" header
    header_match = re.search(r'^-{20,}\n', text, re.MULTILINE)
    if header_match:
        text = text[header_match.end():]

    # Remove common boilerplate phrases
    boilerplate_patterns = [
        r"CAUTION: This email originated from outside of the organization.*?(?=\n\n|\Z)",
        r"Do Not Share This Email.*?(?=\n\n|\Z)",
        r"About Docusign.*?(?=\n\n|\Z)",
        r"Questions about the Document\?.*?(?=\n\n|\Z)",
        r"If you have trouble signing,.*?(?=\n\n|\Z)",
        r"Disclaimer:.*?(?=\n\n|\Z)",
        r"Our Vision:.*?(?=\n\n|\Z)",
        r"If you don't wish to receive our awareness emails,.*?(?=\n\n|\Z)",
        r"This message was sent to you by.*?with your request.",
    ]
    for pattern in boilerplate_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.DOTALL)

    # Remove excessive blank lines and strip
    text = re.sub(r'\n\s*\n', '\n\n', text)
    return text.strip()

def _extract_manifest_metadata(conv: Dict[str, Any]) -> dict:
    """Extract comprehensive metadata from manifest.json including participants"""
    manifest = conv.get("manifest")
    if not isinstance(manifest, dict):
        return {
            "subject": "",
            "end_date": None,
            "start_date": None,
            "conv_id": "",
            "from_email": "",
            "from_name": "",
            "to_recipients": [],
            "cc_recipients": [],
            "to_emails": [],
            "cc_emails": [],
        }

    # Subject and dates
    subject = (manifest.get("smart_subject") or "").strip()
    time_span = manifest.get("time_span", {})
    
    # Extract participants from first message
    messages = manifest.get("messages", [])
    from_info = {}
    to_list = []
    cc_list = []
    
    if messages:
        first_msg = messages[0]
        from_info = first_msg.get("from", {})
        to_list = first_msg.get("to", [])
        cc_list = first_msg.get("cc", [])
    
    # Extract TO recipients (with names and emails)
    to_recipients = []
    for recipient in to_list:
        if isinstance(recipient, dict):
            to_recipients.append({
                "name": recipient.get("name", ""),
                "email": recipient.get("smtp", ""),
            })
    
    # Extract CC recipients (with names and emails)
    cc_recipients = []
    for recipient in cc_list:
        if isinstance(recipient, dict):
            cc_recipients.append({
                "name": recipient.get("name", ""),
                "email": recipient.get("smtp", ""),
            })
    
    return {
        "conv_id": manifest.get("conv_id", ""),
        "subject": subject,
        "start_date": time_span.get("start_local"),
        "end_date": time_span.get("end_local"),
        
        # Sender (with name)
        "from_email": from_info.get("smtp", ""),
        "from_name": from_info.get("name", ""),
        
        # Recipients (with names)
        "to_recipients": to_recipients,
        "cc_recipients": cc_recipients,
        
        # For backward compatibility - just emails
        "to_emails": [r["email"] for r in to_recipients if r["email"]],
        "cc_emails": [r["email"] for r in cc_recipients if r["email"]],
    }

def _build_doc_entries(
    text: Any,
    *,
    doc_id: str,
    doc_path: str,
    subject: str,
    end_date: str | None,
    metadata: dict = {},
    is_attachment: bool = False,
    attachment_info: dict = {},
) -> List[Dict[str, Any]]:
    if not isinstance(text, str):
        return []

    # Apply sanitization to the raw text
    sanitized_text = _sanitize_text(text)
    if not sanitized_text:
        return []

    # Extract conv_id from doc_id
    conv_id = doc_id.split("::")[0]
    
    # Determine doc_type
    doc_type = "attachment" if is_attachment else "conversation"

    if should_chunk_text(sanitized_text, threshold=8000):
        chunks = chunk_for_indexing(
            sanitized_text,
            doc_id=doc_id,
            doc_path=doc_path,
            subject=subject,
            date=end_date,
        )
    else:
        chunks = [
            {
                "id": doc_id,
                "path": doc_path,
                "text": sanitized_text[:200000],
                "subject": subject,
                "date": end_date,
            }
        ]
    
    # Add metadata to each chunk
    for chunk in chunks:
        chunk.update({
            "conv_id": conv_id,
            "doc_type": doc_type,
            "start_date": metadata.get("start_date"),
            "from_email": metadata.get("from_email", ""),
            "from_name": metadata.get("from_name", ""),
            "to_recipients": metadata.get("to_recipients", []),
            "cc_recipients": metadata.get("cc_recipients", []),
            "to_emails": metadata.get("to_emails", []),
            "cc_emails": metadata.get("cc_emails", []),
        })
        
        # Add attachment-specific fields
        if is_attachment and attachment_info:
            chunk.update({
                "attachment_name": attachment_info.get("name", ""),
                "attachment_type": attachment_info.get("extension", ""),
                "attachment_size": attachment_info.get("size", 0),
                "attachment_index": attachment_info.get("index", 0),
            })
    
    return chunks

def _iter_attachment_files(convo_dir: Path) -> Iterable[Path]:
    """
    Lightweight enumeration of attachment files for a conversation directory.
    Mirrors load_conversation's discovery without heavy text extraction.
    """
    attachments: List[Path] = []
    attachments_dir = convo_dir / "Attachments"
    if attachments_dir.exists() and attachments_dir.is_dir():
        for p in attachments_dir.rglob("*"):
            if p.is_file():
                attachments.append(p)
    excluded_files = {"Conversation.txt", "manifest.json", "summary.json"}
    for child in convo_dir.iterdir():
        if child.is_file() and child.name not in excluded_files:
            attachments.append(child)
    return attachments

def build_corpus(root: Path, last_run_time: datetime | None = None, limit: int = 0) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Construct text units, processing only new or modified conversations.
    
    Args:
        root: Root directory containing conversations
        last_run_time: If provided, only process files modified after this time
        limit: If > 0, process only the first `limit` conversations.
        
    Returns:
        Tuple of (new/modified documents, unchanged documents from old index)
    """
    new_or_updated_docs: List[Dict[str, Any]] = []
    unchanged_docs: List[Dict[str, Any]] = []
    
    # Load old mapping if doing incremental
    index_dir = root / INDEX_DIRNAME
    mapping_path = index_dir / MAPPING_NAME
    old_mapping = {}
    processed_ids = set()  # Track which IDs we've processed
    
    if last_run_time and mapping_path.exists():
        try:
            # tolerate BOM if present
            with mapping_path.open('r', encoding='utf-8-sig') as f:
                old_docs = json.load(f)
            for doc in old_docs:
                old_mapping[doc['id']] = doc
        except (json.JSONDecodeError, OSError):
            logger.warning("Could not load old mapping file. Performing a full re-index.")
            last_run_time = None  # Force full re-index

    total_convo_dirs = 0
    processed_convo_dirs = 0

    conversation_dirs = find_conversation_dirs(root)
    if limit > 0:
        logger.info(f"--- Applying limit: processing first {limit} of {len(conversation_dirs)} conversations. ---")
        conversation_dirs = conversation_dirs[:limit]

    for convo_dir in conversation_dirs:
        total_convo_dirs += 1
        convo_txt_path = convo_dir / "Conversation.txt"
        base_id = convo_dir.name
        
        # Check if the conversation needs processing
        needs_processing = True
        if last_run_time and convo_txt_path.exists():
            modified_time = datetime.fromtimestamp(convo_txt_path.stat().st_mtime, tz=timezone.utc)
            if modified_time < last_run_time:
                # Conversation text unchanged. Check attachments individually.
                needs_processing = False

                # Prepare to keep old docs for this conversation, except any attachments that changed.
                convo_prefix = f"{base_id}::"
                base_unchanged = [doc for did, doc in old_mapping.items() if did.startswith(convo_prefix)]

                last_run_ts = last_run_time.timestamp()
                changed_attachment_paths: Set[Path] = set()
                for att_file in _iter_attachment_files(convo_dir):
                    try:
                        if att_file.stat().st_mtime >= last_run_ts:
                            changed_attachment_paths.add(att_file.resolve())
                    except OSError:
                        # If attachment disappeared since last run, treat as changed (it will be removed)
                        changed_attachment_paths.add(att_file.resolve())

                if not changed_attachment_paths:
                    # no attachment changes; keep all old docs
                    for d in base_unchanged:
                        unchanged_docs.append(d)
                        processed_ids.add(d["id"])
                else:
                    # Some attachments changed: rebuild only those; keep others from old mapping
                    conv = load_conversation(convo_dir)
                    # Map path -> attachment index (1-based) using load_conversation ordering
                    path_to_index: Dict[Path, int] = {}
                    for i, attachment in enumerate(conv.get("attachments", []), 1):
                        att_path = Path(attachment.get("path", "")).resolve()
                        if att_path:
                            path_to_index[att_path] = i

                    # Build set of prefixes to exclude from unchanged docs (changed attachments)
                    changed_prefixes: Set[str] = set()
                    for p in changed_attachment_paths:
                        idx = path_to_index.get(p)
                        if idx is not None:
                            changed_prefixes.add(f"{base_id}::att{idx}")

                    # Keep only old docs that are NOT under a changed attachment prefix
                    for d in base_unchanged:
                        did = d["id"]
                        if not any(did == pref or did.startswith(pref + "::") for pref in changed_prefixes):
                            unchanged_docs.append(d)
                            processed_ids.add(did)

                    # Rebuild changed attachments
                    metadata = _extract_manifest_metadata(conv)
                    subject = metadata.get("subject", "")
                    end_date = metadata.get("end_date")
                    for p in changed_attachment_paths:
                        idx = path_to_index.get(p)
                        if idx is None:
                            continue
                        # Find the text for this attachment from load_conversation
                        for i, att in enumerate(conv.get("attachments", []), 1):
                            att_path = Path(att.get("path", "")).resolve()
                            if i == idx and att.get("text"):
                                att_modified_time = p.stat().st_mtime if p.exists() else time.time()
                                attachment_info = {
                                    "name": p.name,
                                    "extension": p.suffix.lower(),
                                    "size": p.stat().st_size if p.exists() else 0,
                                    "index": i,
                                }
                                entry_docs = _build_doc_entries(
                                    att["text"],
                                    doc_id=f"{base_id}::att{i}",
                                    doc_path=str(p),
                                    subject=subject,
                                    end_date=end_date,
                                    metadata=metadata,
                                    is_attachment=True,
                                    attachment_info=attachment_info,
                                )
                                for doc in entry_docs:
                                    doc["modified_time"] = att_modified_time
                                    processed_ids.add(doc["id"])
                                new_or_updated_docs.extend(entry_docs)
                                break

        if needs_processing:
            processed_convo_dirs += 1
            conv = load_conversation(convo_dir)
            
            # Extract FULL metadata
            metadata = _extract_manifest_metadata(conv)
            subject = metadata.get("subject", "")
            end_date = metadata.get("end_date")
            
            # Get modification time
            convo_modified_time = convo_txt_path.stat().st_mtime if convo_txt_path.exists() else time.time()

            convo_text = conv.get("conversation_txt", "")
            if convo_text:
                prepared_text = f"Subject: {subject}\n\n{convo_text}" if subject else convo_text
                doc_entries = _build_doc_entries(
                    prepared_text,
                    doc_id=f"{base_id}::conversation",
                    doc_path=str(convo_dir / "Conversation.txt"),
                    subject=subject,
                    end_date=end_date,
                    metadata=metadata,  # Pass full metadata
                    is_attachment=False,
                )
                # Add modification time to each entry
                for entry in doc_entries:
                    entry["modified_time"] = convo_modified_time
                    processed_ids.add(entry["id"])
                new_or_updated_docs.extend(doc_entries)

            for i, attachment in enumerate(conv.get("attachments", []), 1):
                if isinstance(attachment, dict) and attachment.get("text"):
                    att_path = Path(attachment.get("path", ""))
                    att_modified_time = att_path.stat().st_mtime if att_path.exists() else convo_modified_time
                    
                    # Extract attachment info
                    attachment_info = {
                        "name": att_path.name if att_path else f"attachment_{i}",
                        "extension": att_path.suffix.lower() if att_path else "",
                        "size": att_path.stat().st_size if att_path.exists() else 0,
                        "index": i,
                    }
                    
                    doc_entries = _build_doc_entries(
                        attachment["text"],
                        doc_id=f"{base_id}::att{i}",
                        doc_path=str(att_path),
                        subject=subject,
                        end_date=end_date,
                        metadata=metadata,  # Pass full metadata
                        is_attachment=True,  # Flag as attachment
                        attachment_info=attachment_info,  # Attachment details
                    )
                    # Add modification time to each entry
                    for entry in doc_entries:
                        entry["modified_time"] = att_modified_time
                        processed_ids.add(entry["id"])
                    new_or_updated_docs.extend(doc_entries)

    logger.info(f"Scanned {total_convo_dirs} conversations, processing {processed_convo_dirs} new or updated.")
    return new_or_updated_docs, unchanged_docs

def build_incremental_corpus(root: Path, existing_file_times: Dict[str, float]) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Build corpus incrementally by only processing new or modified files.
    
    Args:
        root: Root directory containing conversations
        existing_file_times: Dictionary of {doc_id: modification_time} from previous index
        
    Returns:
        Tuple of (new/modified documents, list of deleted doc IDs)
    """
    docs: List[Dict[str, Any]] = []
    current_doc_ids = set()
    modified_count = 0
    new_count = 0
    
    for convo_dir in find_conversation_dirs(root):
        conv = load_conversation(convo_dir)
        base_id = Path(conv["path"]).name
        
        # Extract FULL metadata
        metadata = _extract_manifest_metadata(conv)
        subject = metadata.get("subject", "")
        end_date = metadata.get("end_date")
        
        # Use Conversation.txt mtime (directory mtime may not change on file edit)
        convo_txt_path = convo_dir / "Conversation.txt"
        conv_modified_time = convo_txt_path.stat().st_mtime if convo_txt_path.exists() else convo_dir.stat().st_mtime
        conv_doc_id = f"{base_id}::conversation"
        # Ensure we preserve existing chunk IDs as "current"
        for did in list(existing_file_times.keys()):
            if did == conv_doc_id or did.startswith(conv_doc_id + "::"):
                current_doc_ids.add(did)
        
        # Compare against the latest existing time among this conversation's ids (base or chunks)
        existing_time = max(
            (t for did, t in existing_file_times.items() if did == conv_doc_id or did.startswith(conv_doc_id + "::")),
            default=None
        )
        if existing_time is None or conv_modified_time > existing_time:
            if existing_time is None:
                new_count += 1
            else:
                modified_count += 1
                
            convo_text = conv.get("conversation_txt", "")
            if convo_text:
                prepared_text = f"Subject: {subject}\n\n{convo_text}" if subject else convo_text
                entry_docs = _build_doc_entries(
                    prepared_text,
                    doc_id=conv_doc_id,
                    doc_path=str(convo_dir / "Conversation.txt"),
                    subject=subject,
                    end_date=end_date,
                    metadata=metadata,  # Pass full metadata
                    is_attachment=False,
                )
                # Add modification time
                for doc in entry_docs:
                    doc["modified_time"] = conv_modified_time
                docs.extend(entry_docs)

        # Check attachments
        for i, attachment in enumerate(conv.get("attachments", []), 1):
            if isinstance(attachment, dict) and attachment.get("text"):
                att_doc_id = f"{base_id}::att{i}"
                # Mark all existing chunk ids for this attachment as current
                for did in list(existing_file_times.keys()):
                    if did == att_doc_id or did.startswith(att_doc_id + "::"):
                        current_doc_ids.add(did)
                
                # Get attachment file modification time
                att_path = Path(attachment.get("path", ""))
                att_modified_time = conv_modified_time  # Default to conv time
                if att_path.exists():
                    att_modified_time = att_path.stat().st_mtime
                
                # Extract attachment info
                attachment_info = {
                    "name": att_path.name if att_path else f"attachment_{i}",
                    "extension": att_path.suffix.lower() if att_path else "",
                    "size": att_path.stat().st_size if att_path.exists() else 0,
                    "index": i,
                }
                
                # Compare against the latest existing time among this attachment's ids (base or chunks)
                existing_time = max(
                    (t for did, t in existing_file_times.items() if did == att_doc_id or did.startswith(att_doc_id + "::")),
                    default=None
                )
                if existing_time is None or att_modified_time > existing_time:
                    if existing_time is None:
                        new_count += 1
                    else:
                        modified_count += 1
                        
                    entry_docs = _build_doc_entries(
                        attachment["text"],
                        doc_id=att_doc_id,
                        doc_path=str(att_path),
                        subject=subject,
                        end_date=end_date,
                        metadata=metadata,  # Pass full metadata
                        is_attachment=True,  # Flag as attachment
                        attachment_info=attachment_info,  # Attachment details
                    )
                    # Add modification time
                    for doc in entry_docs:
                        doc["modified_time"] = att_modified_time
                    docs.extend(entry_docs)
    
    # Find deleted documents
    deleted_doc_ids = set(existing_file_times.keys()) - current_doc_ids
    
    logger.info(f"Incremental corpus: {new_count} new, {modified_count} modified, {len(deleted_doc_ids)} deleted documents")
    logger.info(f"Total documents to process: {len(docs)}")
    
    return docs, list(deleted_doc_ids)

def load_existing_index(index_dir: Path) -> Tuple[Optional[Any], Optional[List[Dict]], Optional[Dict[str, float]], Optional[np.ndarray]]:
    """
    Load existing FAISS index, mapping, file times, and embeddings if they exist.
    
    Returns:
        Tuple of (faiss_index, mapping_list, file_times_dict, embeddings_array) or (None, None, None, None) if not found
    """
    index_path = index_dir / INDEX_NAME
    mapping_path = index_dir / MAPPING_NAME
    file_times_path = index_dir / FILE_TIMES_NAME
    embeddings_path = index_dir / "embeddings.npy"
    
    if not mapping_path.exists():
        return None, None, None, None
    
    # Load FAISS index (optional)
    index = None
    if HAVE_FAISS and faiss is not None and index_path.exists():
        try:
            index = faiss.read_index(str(index_path))
            logger.info(f"Loaded existing FAISS index with {index.ntotal} vectors")
        except Exception as e:
            logger.warning(f"Failed to load FAISS index: {e}")
    
    # Load mapping
    try:
        # tolerate BOM if present
        with open(mapping_path, 'r', encoding='utf-8-sig') as f:
            mapping = json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load mapping: {e}")
        return None, None, None, None
    
    # Load embeddings (required for incremental updates)
    embeddings = None
    if embeddings_path.exists():
        try:
            embeddings = np.load(str(embeddings_path))
            logger.info(f"Loaded existing embeddings with shape {embeddings.shape}")
        except Exception as e:
            logger.warning(f"Failed to load embeddings: {e}")
    
    # Load file times (optional, may not exist in old indexes)
    file_times = {}
    if file_times_path.exists():
        try:
            with open(file_times_path, 'r', encoding='utf-8') as f:
                file_times = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load file times: {e}")
    
    # Build file times from mapping if not available
    if not file_times and mapping:
        for item in mapping:
            doc_id = item.get("id")
            modified_time = item.get("modified_time")
            if doc_id and modified_time:
                file_times[doc_id] = modified_time
    
    return index, mapping, file_times, embeddings

def save_index(out_dir: Path, embeddings: "np.ndarray", mapping: List[Dict[str, Any]], provider: str, num_folders: int) -> None:
    ensure_dir(out_dir)
    
    # Get the actual dimension from the embeddings
    actual_dimension = embeddings.shape[1]
    
    # Always save NumPy sidecar for portability
    np.save(out_dir / "embeddings.npy", embeddings.astype("float32"))
    
    if HAVE_FAISS and faiss is not None:
        index = faiss.IndexFlatIP(actual_dimension)
        embeddings_float32 = embeddings.astype("float32", order='C')
        index.add(embeddings_float32)
        faiss.write_index(index, str(out_dir / INDEX_NAME))
        
    (out_dir / MAPPING_NAME).write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")
    
    # Save metadata, now including the actual dimension
    metadata = create_index_metadata(
        provider=provider,
        num_documents=len(mapping),
        num_folders=num_folders,
        index_dir=out_dir,
        # Pass the actual dimension to be stored in the metadata
        custom_metadata={"actual_dimensions": actual_dimension}
    )
    save_index_metadata(metadata, out_dir)
    
    logger.info(f"Wrote index with {len(mapping)} items to {out_dir} (dimensions: {actual_dimension})")

def main() -> None:
    # Configure logging for CLI entry point
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    ap = argparse.ArgumentParser(description="Index exported email conversations for retrieval.")
    ap.add_argument("--root", required=True, help="Root folder containing conversation subfolders")
    ap.add_argument("--provider", choices=["vertex", "openai", "azure", "cohere", "huggingface", "local", "qwen"],
                    default=os.getenv("EMBED_PROVIDER", "vertex"),
                    help="Embedding provider. Default from EMBED_PROVIDER env (defaults to vertex).")
    ap.add_argument("--model", default=None,
                    help="Override embedding model for selected provider "
                         "(e.g., vertex: gemini-embedding-001; openai: text-embedding-3-small; "
                         "azure: deployment name)")
    ap.add_argument("--batch", type=int, default=int(os.getenv("EMBED_BATCH","128")), help="Embedding batch size")
    ap.add_argument("--force-reindex", action="store_true", help="Force a full re-indexing of all conversations.")
    ap.add_argument("--limit", type=int, default=0, help="Limit the number of conversations to process (for testing). 0 means no limit.")
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.exists():
        raise SystemExit(f"Root not found: {root}")

    # Ensure model override (if any) is respected by llm_client
    _apply_model_override(args.provider, args.model)

    out_dir = root / INDEX_DIRNAME
    last_run_time = None if args.force_reindex else _get_last_run_time(out_dir)
    
    if last_run_time and not args.force_reindex:
        logger.info(f"Starting incremental index update from {last_run_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    else:
        logger.info("No previous run time found or --force-reindex used. Starting a full index build.")

    # Try to load existing index for incremental updates
    existing_index, existing_mapping, existing_file_times, existing_embeddings = load_existing_index(out_dir)
    
    # Use incremental corpus building when file times are available
    if existing_file_times and not args.force_reindex:
        # Use the per-file incremental approach
        new_docs, deleted_ids = build_incremental_corpus(root, existing_file_times)
        
        # Get unchanged documents from existing mapping
        deleted_ids_set = set(deleted_ids)
        new_doc_ids = {doc["id"] for doc in new_docs}
        unchanged_docs = [
            doc for doc in (existing_mapping or [])
            if doc["id"] not in new_doc_ids and doc["id"] not in deleted_ids_set
        ]
        
        if deleted_ids:
            logger.info(f"Removing {len(deleted_ids)} deleted documents from index")
            
    else:
        # Fall back to the original corpus building for first run or force reindex
        new_docs, unchanged_docs = build_corpus(root, last_run_time, limit=args.limit)

    if not new_docs and unchanged_docs:
        logger.info("No new or updated conversations to index. Index is already up to date.")
        _save_run_time(out_dir)
        return
        
    if not new_docs and not unchanged_docs:
        raise SystemExit("No conversations found to index.")

    # For incremental indexing: reuse embeddings for unchanged docs
    if last_run_time and existing_embeddings is not None and existing_mapping:
        # Create a mapping of doc_id to embedding index
        existing_id_to_idx = {doc["id"]: idx for idx, doc in enumerate(existing_mapping)}
        
        # Separate unchanged docs that have embeddings vs those that don't
        unchanged_with_embeddings = []
        unchanged_embeddings_list = []
        unchanged_without_embeddings = []
        
        for doc in unchanged_docs:
            doc_id = doc["id"]
            if doc_id in existing_id_to_idx:
                idx = existing_id_to_idx[doc_id]
                if idx < len(existing_embeddings):
                    unchanged_with_embeddings.append(doc)
                    unchanged_embeddings_list.append(existing_embeddings[idx])
                else:
                    unchanged_without_embeddings.append(doc)
            else:
                unchanged_without_embeddings.append(doc)
        
        logger.info(f"Reusing embeddings for {len(unchanged_with_embeddings)} unchanged documents")
        
        # Only embed new documents and unchanged docs without embeddings
        docs_to_embed = new_docs + unchanged_without_embeddings
        
        if docs_to_embed:
            texts = [d.get("text", "") for d in docs_to_embed]
            valid_indices = [i for i, t in enumerate(texts) if t and t.strip()]
            valid_texts = [texts[i] for i in valid_indices]
            valid_docs = [docs_to_embed[i] for i in valid_indices]
            
            if valid_texts:
                logger.info(f"Embedding {len(valid_texts)} new/modified documents...")
                
                # Embed texts in batches
                new_embeddings = []
                batch_size = args.batch
                for i in range(0, len(valid_texts), batch_size):
                    batch_texts = valid_texts[i:i+batch_size]
                    batch_embeddings = embed_texts(batch_texts, provider=args.provider)
                    new_embeddings.append(batch_embeddings)
                    logger.info(f"  ... processed {i + len(batch_texts)} / {len(valid_texts)}")
                
                if new_embeddings:
                    new_embs = np.vstack(new_embeddings)
                    
                    # Combine unchanged embeddings with new embeddings
                    if unchanged_embeddings_list:
                        unchanged_embs = np.vstack(unchanged_embeddings_list)
                        embs = np.vstack([unchanged_embs, new_embs])
                        final_docs = unchanged_with_embeddings + valid_docs
                    else:
                        embs = new_embs
                        final_docs = valid_docs
                else:
                    # If embedding failed but we have unchanged docs, use those
                    if unchanged_with_embeddings:
                        embs = np.vstack(unchanged_embeddings_list)
                        final_docs = unchanged_with_embeddings
                    else:
                        raise SystemExit("Embedding failed for all documents.")
            else:
                # No valid texts to embed, just use unchanged
                if unchanged_with_embeddings:
                    embs = np.vstack(unchanged_embeddings_list)
                    final_docs = unchanged_with_embeddings
                else:
                    logger.info("No documents to index after filtering.")
                    _save_run_time(out_dir)
                    return
        else:
            # Only unchanged documents with embeddings
            embs = np.vstack(unchanged_embeddings_list)
            final_docs = unchanged_with_embeddings
    else:
        # Full re-index: embed everything
        final_docs = unchanged_docs + new_docs
        texts = [d.get("text", "") for d in final_docs]
        
        # Filter out any empty texts
        valid_texts = [t for t in texts if t and t.strip()]
        if len(valid_texts) < len(final_docs):
            logger.warning(f"Filtered out {len(final_docs) - len(valid_texts)} empty documents.")
            final_docs = [doc for doc in final_docs if doc.get("text", "").strip()]

        if not final_docs:
            logger.info("No documents to index after filtering.")
            _save_run_time(out_dir)
            return

        logger.info(f"Embedding {len(final_docs)} documents...")

        # Embed texts in batches
        all_embeddings = []
        batch_size = args.batch
        for i in range(0, len(valid_texts), batch_size):
            batch_texts = valid_texts[i:i+batch_size]
            batch_embeddings = embed_texts(batch_texts, provider=args.provider)
            all_embeddings.append(batch_embeddings)
            logger.info(f"  ... processed {i + len(batch_texts)} / {len(valid_texts)}")

        if not all_embeddings:
            raise SystemExit("Embedding failed for all documents.")

        embs = np.vstack(all_embeddings)
    # Sanity check alignment between embeddings and documents
    if embs.shape[0] != len(final_docs):
        raise SystemExit(f"Embedding/document count mismatch: embeddings={embs.shape[0]} docs={len(final_docs)}")

    unique_folders = len({d["id"].split("::")[0] for d in final_docs})
    logger.info(f"Final index will contain {len(final_docs)} documents from {unique_folders} folders")

    # Save the final index with ALL metadata fields
    mapping_data = []
    file_times = {}
    for d in final_docs:
        doc_entry = {
            "id": d["id"],
            "conv_id": d.get("conv_id", ""),
            "doc_type": d.get("doc_type", "conversation"),
            "path": d["path"],
            "subject": d.get("subject", ""),
            "date": d.get("date"),
            "start_date": d.get("start_date"),
            "snippet": d.get("text", "")[:500],
            "modified_time": d.get("modified_time", time.time()),
            
            # Participant fields
            "from_email": d.get("from_email", ""),
            "from_name": d.get("from_name", ""),
            "to_recipients": d.get("to_recipients", []),
            "cc_recipients": d.get("cc_recipients", []),
            "to_emails": d.get("to_emails", []),
            "cc_emails": d.get("cc_emails", []),
            
            # Attachment fields (if applicable)
            "attachment_name": d.get("attachment_name", ""),
            "attachment_type": d.get("attachment_type", ""),
            "attachment_size": d.get("attachment_size", 0),
            "attachment_index": d.get("attachment_index", 0),
        }
        mapping_data.append(doc_entry)
        file_times[d["id"]] = d.get("modified_time", time.time())
    
    save_index(out_dir, embs, mapping_data, provider=args.provider, num_folders=unique_folders)
    
    # Save file times for next incremental run
    (out_dir / FILE_TIMES_NAME).write_text(
        json.dumps(file_times, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    
    _save_run_time(out_dir)

if __name__ == "__main__":
    main()
