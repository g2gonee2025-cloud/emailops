from __future__ import annotations

import contextlib
import hashlib
import logging
import mimetypes
from collections.abc import Mapping
from pathlib import Path
from typing import Any, TypedDict, cast

from emailops.core_config import get_config
from emailops.core_text_extraction import extract_text

from .mapitags import (
    PR_ATTACH_CONTENT_ID_A,
    PR_ATTACH_CONTENT_ID_W,
    PR_ATTACH_DATA_BIN,
    PR_ATTACH_FLAGS,
    PR_ATTACH_LAST_MODIFICATION_TIME,
    PR_ATTACHMENT_HIDDEN,
    PR_LAST_MODIFICATION_TIME,
)
from .utils import OL_FORMAT_HTML, ensure_dir, sanitize_filename, to_iso_z

# Ensure .msg has a sensible MIME type
mimetypes.add_type("application/vnd.ms-outlook", ".msg", strict=False)

log = logging.getLogger(__name__)


class AttachmentEntry(TypedDict):
    filename: str
    filetype: str
    size: int
    hash: str
    source_modified: str
    extracted_text: str


class FailedAttachment(TypedDict):
    filename: str
    reason: str
    size: int

def _guess_mime_type(filename: str) -> str:
    typ, _ = mimetypes.guess_type(filename)
    return typ or "application/octet-stream"

def _adler32_head_tail(path: Path, head: int = 4096, tail: int = 4096) -> str:
    """Adler32 of (head+tail) without loading the entire file into memory."""
    import zlib
    try:
        size = path.stat().st_size
        with path.open("rb") as f:
            head_bytes = f.read(head)
            if size > head + tail:
                f.seek(max(size - tail, 0))
                tail_bytes = f.read(tail)
                sample = head_bytes + tail_bytes
            else:
                f.seek(0)
                sample = f.read()
        return "adler:" + f"{(zlib.adler32(sample) & 0xFFFFFFFF):08X}"
    except Exception:
        return "adler:00000000"

def _unique_path(base_dir: Path, filename: str) -> Path:
    """Return a unique path within base_dir, avoiding collisions and too-long names.

    Security: Validates path to prevent traversal attacks by ensuring final path
    is within base_dir and uses absolute path resolution.
    """
    base = sanitize_filename(filename) or "attachment"
    base_dir_resolved = base_dir.resolve()
    candidate = base_dir / base

    # Security: Resolve to absolute path and verify it's within base_dir
    try:
        candidate_resolved = candidate.resolve()

        # Check if candidate is within base_dir
        if not str(candidate_resolved).startswith(str(base_dir_resolved)):
            raise ValueError(f"Path traversal detected: {filename}")
    except Exception as e:
        log.error("Path validation failed for %s: %s", filename, e)
        # Use safe fallback name
        safe_name = hashlib.md5(base.encode()).hexdigest()[:16] + ".dat"
        candidate = base_dir / safe_name
        candidate_resolved = candidate.resolve()

    if not candidate.exists() and len(str(candidate_resolved)) < 250:
        return candidate_resolved

    stem = candidate.stem
    suffix = candidate.suffix
    i = 1
    while True:
        cand = base_dir / f"{stem} ({i}){suffix}"
        try:
            cand_resolved = cand.resolve()
            # Verify still within base_dir
            if not str(cand_resolved).startswith(str(base_dir_resolved)):
                raise ValueError(f"Path traversal in numbered variant: {i}")
            if len(str(cand_resolved)) < 250 and not cand.exists():
                return cand_resolved
        except Exception:
            pass
        i += 1
        if i > 200:
            # fallback hash name
            safe_name = hashlib.md5(base.encode()).hexdigest()[:16] + suffix
            return (base_dir / safe_name).resolve()

def _prop_to_iso_z(pa: Any, mapi_tag: str) -> str:
    try:
        if pa:
            dt = pa.GetProperty(mapi_tag)
            return to_iso_z(dt)
    except Exception:
        pass
    return ""

def _attachment_stable_hash(attachment: Any) -> str | None:
    """Compute a stable SHA-256 of the raw MAPI payload (PR_ATTACH_DATA_BIN)."""
    try:
        pa = getattr(attachment, "PropertyAccessor", None)
        if not pa:
            log.debug("PropertyAccessor unavailable for attachment hash computation")
            return None
        blob = pa.GetProperty(PR_ATTACH_DATA_BIN)
        if blob is None:
            log.debug("PR_ATTACH_DATA_BIN property is None for attachment")
            return None
        data = bytes(blob)
        hash_val = "sha256:" + hashlib.sha256(data).hexdigest()
        log.debug("Computed stable hash for attachment (%d bytes)", len(data))
        return hash_val
    except Exception as e:
        log.debug("Failed to compute stable hash: %s", e)
        return None

_INLINE_FLAG = 0x4

def _validate_saved_file(path: Path, original_name: str) -> bool:
    """Validate saved attachment file for corruption and expected MIME type.

    Returns: True if file passes validation, False otherwise
    """
    try:
        # Check file exists and has content
        if not path.exists():
            log.error("Saved file does not exist: %s", path)
            return False

        size = path.stat().st_size
        if size == 0:
            log.error("Saved file has zero size: %s", path)
            return False

        # Verify MIME type matches extension
        guessed_type = _guess_mime_type(path.name)
        original_type = _guess_mime_type(original_name)

        # If types differ significantly, log warning but don't fail
        # (Outlook may change extensions during save)
        if guessed_type != original_type:
            log.debug(
                "MIME type mismatch for %s: saved as %s, expected %s",
                original_name, guessed_type, original_type
            )

        # Try to read first few bytes to ensure file isn't corrupted
        try:
            with path.open('rb') as f:
                header = f.read(min(1024, size))
                if len(header) == 0:
                    log.error("Cannot read file content: %s", path)
                    return False
        except Exception as e:
            log.error("File read validation failed for %s: %s", path, e)
            return False

        return True
    except Exception as e:
        log.error("File validation error for %s: %s", path, e)
        return False

def _is_inline(attachment: Any, mail_item: Any) -> bool:
    """Heuristic to detect inline/hidden/signature attachments (skip saving)."""
    try:
        pa = getattr(attachment, "PropertyAccessor", None)
        if pa is None:
            return False
        # Hidden
        try:
            if bool(pa.GetProperty(PR_ATTACHMENT_HIDDEN)):
                return True
        except Exception:
            pass
        # Inline flag
        try:
            flags = pa.GetProperty(PR_ATTACH_FLAGS)
            if flags and (int(flags) & _INLINE_FLAG) != 0:
                return True
        except Exception:
            pass
        # Content-ID reference (CID) present in HTML body
        content_id = None
        try:
            content_id = pa.GetProperty(PR_ATTACH_CONTENT_ID_W)
        except Exception:
            try:
                content_id = pa.GetProperty(PR_ATTACH_CONTENT_ID_A)
            except Exception:
                content_id = None
        if content_id:
            try:
                if getattr(mail_item, "BodyFormat", 0) == OL_FORMAT_HTML:
                    html = (getattr(mail_item, "HTMLBody", "") or "").lower()
                    cid = str(content_id).lower()
                    if f"cid:{cid}" in html or f"cid:<{cid}>" in html:
                        return True
            except Exception:
                pass
        # Tiny signature/branding images even without CID
        name = (getattr(attachment, "FileName", "") or "").lower()
        try:
            size = int(getattr(attachment, "Size", 0) or 0)
        except Exception:
            size = 0
        if name.endswith((".png", ".gif", ".jpg", ".jpeg")) and size < 50 * 1024:
            return any(tok in name for tok in ("sig", "signature", "logo", "pixel", "spacer"))
    except Exception:
        return False
    return False

def save_attachments_for_items(
    mail_items: list[Any],
    conv_dir: Path,
    max_file_size_mb: float | None = None,
    max_total_size_mb: float | None = None,
    blocked_extensions: set[str] | None = None,
    previous_manifest: Mapping[str, Any] | None = None,
    extract_attachment_text: bool = False
) -> tuple[bool, list[AttachmentEntry]]:
    """Save non-inline attachments and optionally extract text."""
    config = get_config()

    if max_file_size_mb is not None:
        file_size_limit: float | None = max_file_size_mb
    else:
        configured_limit = getattr(config.limits, "skip_attachment_over_mb", None)
        file_size_limit = (
            float(configured_limit)
            if isinstance(configured_limit, (int, float))
            else None
        )

    if max_total_size_mb is not None:
        total_size_limit: float | None = max_total_size_mb
    else:
        configured_total_limit = getattr(config.limits, "max_total_attachments_mb", None)
        total_size_limit = (
            float(configured_total_limit)
            if isinstance(configured_total_limit, (int, float))
            else None
        )
    if blocked_extensions is None:
        blocked_extensions = set(config.security.blocked_extensions)

    attachments_dir = conv_dir / "Attachments"
    ensure_dir(attachments_dir)

    meta: list[AttachmentEntry] = []
    total_size = 0
    count = 0
    failed_attachments: list[FailedAttachment] = []

    # Build index of previously exported attachments from manifest
    local_index: dict[str, AttachmentEntry] = {}
    filename_index: dict[str, str] = {}
    if isinstance(previous_manifest, Mapping):
        attachments_val = previous_manifest.get("attachments")
        if isinstance(attachments_val, list):
            for att_obj in cast(list[Any], attachments_val):
                if not isinstance(att_obj, Mapping):
                    continue
                att = cast(Mapping[str, Any], att_obj)
                hash_obj = att.get("hash")
                filename_obj = att.get("filename")
                if not isinstance(hash_obj, str) or not isinstance(filename_obj, str):
                    continue
                size_obj = att.get("size", 0)
                try:
                    size_val = int(size_obj)  # Defensive cast for persisted manifests
                except (TypeError, ValueError):
                    size_val = 0
                filetype_obj = att.get("filetype", "")
                source_modified_obj = att.get("source_modified", "")
                extracted_text_obj = att.get("extracted_text", "")
                manifest_entry: AttachmentEntry = {
                    "filename": filename_obj,
                    "filetype": str(filetype_obj) if filetype_obj is not None else "",
                    "size": size_val,
                    "hash": hash_obj,
                    "source_modified": (
                        str(source_modified_obj)
                        if source_modified_obj is not None
                        else ""
                    ),
                    "extracted_text": (
                        str(extracted_text_obj)
                        if extracted_text_obj is not None
                        else ""
                    ),
                }
                local_index[hash_obj] = manifest_entry
                filename_index[filename_obj] = hash_obj

    for it in mail_items:
        try:
            for a in it.Attachments:
                raw_name = ""
                try:
                    raw_name = getattr(a, "FileName", "") or "unnamed"

                    if _is_inline(a, it):
                        log.debug("Skipping inline attachment: %s", raw_name)
                        continue

                    # Check extension
                    ext = Path(raw_name).suffix.lower()
                    if ext in blocked_extensions:
                        log.warning("Skipping blocked attachment type: %s", raw_name)
                        failed_attachments.append({"filename": raw_name, "reason": "Blocked extension", "size": 0})
                        continue

                    # Check size
                    size = int(getattr(a, "Size", 0) or 0)
                    if file_size_limit is not None and (size / (1024 * 1024)) > file_size_limit:
                        log.warning("Skipping large attachment: %s (%.2f MB)", raw_name, size / (1024*1024))
                        failed_attachments.append({"filename": raw_name, "reason": "Exceeds size limit", "size": size})
                        continue

                    if total_size_limit is not None and ((total_size + size) / (1024 * 1024)) > total_size_limit:
                        log.warning("Total attachment size limit reached, skipping: %s", raw_name)
                        failed_attachments.append({"filename": raw_name, "reason": "Total size limit reached", "size": size})
                        continue

                    # Check for duplicates using stable hash
                    stable_hash = _attachment_stable_hash(a)
                    if stable_hash and stable_hash in local_index:
                        log.debug("Skipping duplicate attachment with stable hash: %s", raw_name)
                        meta.append(local_index[stable_hash])
                        count += 1
                        total_size += size
                        continue

                    # Deduplicate by name if hash is missing
                    safe_name = sanitize_filename(raw_name)
                    if not stable_hash and safe_name in filename_index:
                        log.debug("Skipping duplicate attachment by filename: %s", raw_name)
                        hash_val: str | None = filename_index.get(safe_name)
                        if hash_val and hash_val in local_index:
                            meta.append(local_index[hash_val])
                            count += 1
                            total_size += size
                            continue

                    path = _unique_path(attachments_dir, raw_name)
                    a.SaveAsFile(str(path))

                    if not _validate_saved_file(path, raw_name):
                        log.error("Attachment failed validation after save: %s", raw_name)
                        failed_attachments.append({"filename": raw_name, "reason": "Failed validation", "size": size})
                        with contextlib.suppress(OSError):
                            path.unlink()
                        continue

                    total_size += size
                    file_hash: str = stable_hash or _adler32_head_tail(path)

                    extracted_text = ""
                    if extract_attachment_text:
                        try:
                            extracted_text = extract_text(path)
                        except Exception as e:
                            log.warning(f"Failed to extract text from {path.name}: {e}")

                    source_modified_value = (
                        _prop_to_iso_z(
                            getattr(a, "PropertyAccessor", None),
                            PR_ATTACH_LAST_MODIFICATION_TIME,
                        )
                        or _prop_to_iso_z(
                            getattr(it, "PropertyAccessor", None),
                            PR_LAST_MODIFICATION_TIME,
                        )
                        or to_iso_z(getattr(it, "ReceivedTime", None))
                        or ""
                    )

                    entry: AttachmentEntry = {
                        "filename": path.name,
                        "filetype": _guess_mime_type(path.name),
                        "size": size,
                        "hash": file_hash,
                        "source_modified": source_modified_value,
                        "extracted_text": extracted_text
                    }
                    meta.append(entry)
                    if stable_hash:
                        local_index[stable_hash] = entry
                        filename_index[safe_name] = stable_hash
                    count += 1
                except Exception as e:
                    log.warning("Attachment processing error: %s", e)
                    failed_attachments.append({
                        "filename": raw_name if 'raw_name' in locals() else "unknown",
                        "reason": f"Processing error: {e}",
                        "size": 0
                    })
                    continue
        except Exception:
            continue

    # Log summary
    if failed_attachments:
        log.warning(
            "Conversation had %d failed attachments out of %d total",
            len(failed_attachments), count + len(failed_attachments)
        )
        for fail in failed_attachments[:5]:  # Log first 5 failures
            log.debug("Failed: %s - %s", fail["filename"], fail["reason"])

    return (count > 0, meta)
