"""
Parses attachments_log.csv for rich attachment metadata.
"""

import csv
import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, TextIO

logger = logging.getLogger(__name__)


def _is_within_base(base_dir: Path, candidate: Path) -> bool:
    try:
        candidate.relative_to(base_dir)
    except ValueError:
        return False
    return True


def _open_csv_candidate(base_dir: Path, candidate: Path) -> tuple[Path, TextIO] | None:
    try:
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError:
        return None

    if not _is_within_base(base_dir, resolved):
        raise ValueError(
            "Path traversal attempt detected: "
            f"Attachments log '{resolved}' is outside of secure upload directory '{base_dir}'."
        )

    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(resolved, flags)
    except FileNotFoundError:
        return None
    return resolved, open(fd, encoding="utf-8-sig", newline="", closefd=True)


def parse_attachments_log(
    upload_dir: Path, convo_id: str
) -> dict[str, list[dict[str, Any]]]:
    """
    Parse attachments_log.csv and return a mapping of filename -> metadata.

    Args:
        upload_dir: The secure base directory for uploads.
        convo_id: The identifier for the conversation, used to construct the path.

    Returns:
        Dict mapping filename to a list of metadata dicts (one per occurrence) including:
        - sender
        - mail_time
        - mail_subject
        - attachment_kind (inline vs attachment)
        If a filename appears multiple times in the log, all entries are preserved under that filename key.
    """
    base_dir = upload_dir.resolve()
    convo_dir = (base_dir / convo_id).resolve()

    # Security check: Ensure convo_dir is within the secure base_dir
    if base_dir not in convo_dir.parents and base_dir != convo_dir:
        raise ValueError(
            "Path traversal attempt detected: "
            f"Conversation directory '{convo_dir}' is outside of the secure upload directory '{base_dir}'."
        )

    # Check potential locations (Outlook/folder/attachments/attachments_log.csv)
    candidates = [
        convo_dir / "attachments" / "attachments_log.csv",
        convo_dir / "Attachments" / "attachments_log.csv",
    ]

    metadata_map = defaultdict(list)
    opened: tuple[Path, TextIO] | None = None
    for candidate in candidates:
        opened = _open_csv_candidate(base_dir, candidate)
        if opened:
            break
    if not opened:
        return {}

    csv_path, file_handle = opened
    try:
        with file_handle as f:
            reader = csv.DictReader(f)
            for row in reader:
                filename = row.get("filename")
                if not filename:
                    continue

                # Store relevant fields
                metadata = {
                    "sender": row.get("sender"),
                    "mail_subject": row.get("mail_subject"),
                    "mail_time": row.get("mail_time"),
                    "attachment_kind": row.get("attachment_kind"),
                    "mail_entryid": row.get("mail_entryid"),
                }
                metadata_map[filename].append(metadata)

        logger.info(
            "Loaded metadata for %s unique attachment filenames from %s",
            len(metadata_map),
            csv_path.name,
        )
    except (OSError, csv.Error, UnicodeDecodeError):
        logger.exception("Failed to parse CSV file at %s", csv_path)
        raise

    return dict(metadata_map)
