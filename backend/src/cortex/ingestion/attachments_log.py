"""
Parses attachments_log.csv for rich attachment metadata.
"""
import csv
import logging
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)


def parse_attachments_log(convo_dir: Path) -> Dict[str, Dict[str, Any]]:
    """
    Parse attachments_log.csv and return a mapping of filename -> metadata.

    Returns:
        Dict mapping filename to metadata dict including:
        - sender
        - mail_time
        - mail_subject
        - attachment_kind (inline vs attachment)
    """
    # Check potential locations (Outlook/folder/attachments/attachments_log.csv)
    candidates = [
        convo_dir / "attachments" / "attachments_log.csv",
        convo_dir / "Attachments" / "attachments_log.csv",
    ]

    csv_path = next((p for p in candidates if p.exists()), None)
    if not csv_path:
        return {}

    metadata_map = {}
    try:
        with open(csv_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                filename = row.get("filename")
                if not filename:
                    continue

                # Store relevant fields
                metadata_map[filename] = {
                    "sender": row.get("sender"),
                    "mail_subject": row.get("mail_subject"),
                    "mail_time": row.get("mail_time"),
                    "attachment_kind": row.get("attachment_kind"),
                    "mail_entryid": row.get("mail_entryid"),
                }

        logger.info(
            f"Loaded metadata for {len(metadata_map)} attachments from {csv_path.name}"
        )
    except Exception as e:
        logger.warning(f"Failed to parse {csv_path}: {e}")

    return metadata_map
