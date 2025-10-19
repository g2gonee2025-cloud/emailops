from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import hjson  # type: ignore

from .core_text_extraction import extract_text
from .util_files import read_text_file, scrub_json
from .util_processing import get_processing_config

"""
Conversation loading utilities.
Handles loading conversation content, manifest/summary JSON, and attachment texts.
"""

logger = logging.getLogger(__name__)

# Control character pattern
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]")


def load_conversation(
    convo_dir: Path,
    include_attachment_text: bool = False,
    max_total_attachment_text: int | None = None,
    *,
    max_attachment_text_chars: int | None = None,
    skip_if_attachment_over_mb: float | None = None,
) -> dict[str, Any]:
    """
    Load conversation content, manifest/summary JSON, and attachment texts.

    Returns:
        dict with keys: path, conversation_txt, attachments, summary, manifest
    """
    # Get configuration with defaults
    config = get_processing_config()
    if max_total_attachment_text is None:
        max_total_attachment_text = config.max_total_attachment_text
    if max_attachment_text_chars is None:
        max_attachment_text_chars = config.max_attachment_chars
    if skip_if_attachment_over_mb is None:
        skip_if_attachment_over_mb = config.skip_attachment_over_mb

    convo_txt_path = convo_dir / "Conversation.txt"
    summary_json = convo_dir / "summary.json"
    manifest_json = convo_dir / "manifest.json"

    # Read conversation text with BOM handling and sanitation
    conversation_text = ""
    if convo_txt_path.exists():
        try:
            conversation_text = read_text_file(convo_txt_path)
        except Exception as e:
            logger.warning("Failed to read Conversation.txt at %s: %s", convo_dir, e)
            conversation_text = ""

    conv: dict[str, Any] = {
        "path": str(convo_dir),
        "conversation_txt": conversation_text,
        "attachments": [],
        "summary": {},
        "manifest": {},
    }

    # Load manifest.json with strict → repaired → hjson fallback
    if manifest_json.exists():
        conv["manifest"] = scrub_json(_load_manifest_json(manifest_json, convo_dir))

    # Build attachment file list (avoid duplicates)
    attachment_files = _collect_attachment_files(convo_dir)

    # Process attachments
    total_appended = 0
    for att_file in attachment_files:
        try:
            if skip_if_attachment_over_mb and skip_if_attachment_over_mb > 0:
                try:
                    mb = att_file.stat().st_size / (1024 * 1024)
                    if mb > skip_if_attachment_over_mb:
                        logger.info(
                            "Skipping large attachment (%.2f MB > %.2f MB): %s",
                            mb,
                            skip_if_attachment_over_mb,
                            att_file,
                        )
                        continue
                except Exception:
                    pass

            txt = extract_text(att_file, max_chars=max_attachment_text_chars)
            if txt.strip():
                att_rec = {"path": str(att_file), "text": txt}
                conv["attachments"].append(att_rec)

                # Optionally append a truncated view of the attachment into conversation_txt
                if include_attachment_text and total_appended < max_total_attachment_text:
                    remaining = max_total_attachment_text - total_appended
                    # Use relative path header for clarity and determinism
                    try:
                        rel = att_file.relative_to(convo_dir)
                    except Exception:
                        rel = att_file.name
                    header = f"\n\n--- ATTACHMENT: {rel} ---\n\n"
                    snippet = txt[: max(0, remaining - len(header))]
                    conv["conversation_txt"] += header + snippet
                    total_appended += len(header) + len(snippet)
        except Exception as e:
            logger.warning("Failed to process attachment %s: %s", att_file, e)

    # Load summary.json (BOM-safe)
    if summary_json.exists():
        try:
            conv["summary"] = scrub_json(json.loads(summary_json.read_text(encoding="utf-8-sig")))
        except Exception:
            conv["summary"] = {}

    return conv


def _load_manifest_json(manifest_json: Path, convo_dir: Path) -> dict[str, Any]:
    """Load manifest.json with fallback parsing strategies."""
    raw_text = ""
    try:
        raw_bytes = manifest_json.read_bytes()
        try:
            raw_text = raw_bytes.decode("utf-8-sig")
        except UnicodeDecodeError:
            raw_text = raw_bytes.decode("latin-1", errors="ignore")
            logger.warning("Manifest at %s was not valid UTF-8, fell back to latin-1.", convo_dir)

        # Aggressive sanitization to catch a wider range of control chars.
        sanitized = _CONTROL_CHARS.sub("", raw_text)

        # 1) Try strict JSON first (no repair)
        try:
            return json.loads(sanitized)
        except json.JSONDecodeError:
            # 2) Apply backslash repair then try JSON again
            repaired = re.sub(r'(?<!\\)\\(?!["\\/bfnrtu])', r"\\\\", sanitized)
            try:
                return json.loads(repaired)
            except json.JSONDecodeError as e2:
                # 3) Try HJSON on the repaired string
                logger.warning(
                    "Failed strict JSON parse for manifest at %s: %s. Attempting HJSON.",
                    convo_dir,
                    e2,
                )
                try:

                    result = hjson.loads(repaired)
                    logger.info("Successfully parsed manifest for %s using hjson.", convo_dir)
                    return result
                except Exception as hjson_e:
                    logger.error(
                        "hjson also failed to parse manifest for %s: %s. Using empty manifest.",
                        convo_dir,
                        hjson_e,
                    )
                    return {}
    except Exception as e:
        logger.warning(
            "Unexpected error while loading manifest from %s: %s. Skipping.",
            convo_dir,
            e,
        )
        return {}


def _collect_attachment_files(convo_dir: Path) -> list[Path]:
    """
    Collect attachment files, avoiding duplicates and sorting deterministically.

    Note: This has inherent TOCTOU (time-of-check-to-time-of-use) race conditions
    since files could be added/removed between directory listing and actual use.
    We handle this gracefully by catching exceptions during file access.
    """
    attachment_files: list[Path] = []

    # HIGH #10: Improved error handling for attachment directory iteration
    attachments_dir = convo_dir / "Attachments"
    if attachments_dir.exists() and attachments_dir.is_dir():
        try:
            # Collect files but handle potential race conditions during iteration
            attachment_files.extend([p for p in attachments_dir.rglob("*") if p.is_file()])
        except (OSError, PermissionError) as e:
            logger.warning("Failed to iterate Attachments directory at %s: %s", attachments_dir, e)
        except Exception as e:
            logger.error("Unexpected error iterating Attachments directory at %s: %s", attachments_dir, e)

    excluded = {"Conversation.txt", "manifest.json", "summary.json"}
    try:
        # HIGH #10: Snapshot directory contents to minimize race window
        # Convert iterator to list immediately to reduce TOCTOU window
        children = list(convo_dir.iterdir())
        for child in children:
            try:
                # Validate each file individually with error handling
                if child.is_file() and child.name not in excluded:
                    attachment_files.append(child)
            except (OSError, PermissionError) as e:
                # File may have been deleted between listing and checking
                logger.debug("Skipping file that became inaccessible: %s - %s", child, e)
                continue
    except (OSError, PermissionError) as e:
        logger.warning("Failed to iterate conversation dir %s: %s", convo_dir, e)
    except Exception as e:
        logger.error("Unexpected error iterating conversation dir %s: %s", convo_dir, e)

    # Deduplicate then sort deterministically
    seen: set[str] = set()
    unique_files: list[Path] = []
    for f in attachment_files:
        try:
            # HIGH #10: Validate file still exists before resolving
            if not f.exists():
                logger.debug("Skipping attachment that no longer exists: %s", f)
                continue
            s = str(f.resolve())
        except (OSError, PermissionError) as e:
            # File may have been deleted/moved during processing
            logger.debug("Skipping attachment that became inaccessible during resolution: %s - %s", f, e)
            continue
        except Exception as e:
            logger.warning("Failed to resolve attachment path %s: %s", f, e)
            s = str(f)
        if s not in seen:
            seen.add(s)
            unique_files.append(f)

    unique_files.sort(key=lambda p: (p.parent.as_posix(), p.name.lower()))

    return unique_files
