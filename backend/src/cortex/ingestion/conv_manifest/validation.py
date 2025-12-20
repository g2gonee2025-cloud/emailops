"""
Export Validation & Manifest Refresh (B1).

Implements §5 of the Canonical Blueprint.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from cortex.ingestion.core_manifest import load_manifest
from cortex.utils.atomic_io import atomic_write_json
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class Problem(BaseModel):
    folder: str
    issue: str


class ManifestValidationReport(BaseModel):
    schema_version: Dict[str, str] = Field(
        default_factory=lambda: {"id": "manifest_validation_report", "version": "1"},
        alias="schema",
    )
    root: str
    folders_scanned: int = 0
    manifests_created: int = 0
    manifests_updated: int = 0
    problems: List[Problem] = Field(default_factory=list)


def scan_and_refresh(root: Path) -> ManifestValidationReport:
    """
    Scan export directory and ensure manifests are valid and consistent.

    Args:
        root: Path to mail_export directory

    Returns:
        ManifestValidationReport with statistics and problems
    """
    report = ManifestValidationReport(root=str(root.absolute()))

    if not root.exists():
        logger.error(f"Export root not found: {root}")
        return report

    # Ensure artifacts directory exists
    artifacts_dir = root.parent / "artifacts" / "B1_manifests"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Scan conversation folders
    for conv_dir in root.iterdir():
        if not conv_dir.is_dir():
            continue

        conv_txt = conv_dir / "Conversation.txt"
        if not conv_txt.exists():
            continue

        report.folders_scanned += 1
        folder_rel = conv_dir.name

        # Check/Create attachments dir
        attachments_dir = conv_dir / "attachments"
        if not attachments_dir.exists():
            attachments_dir.mkdir(exist_ok=True)
            report.problems.append(
                Problem(folder=folder_rel, issue="attachments_dir_created")
            )

        # Calculate SHA256 of Conversation.txt (normalized)
        try:
            sha256 = _calculate_conversation_hash(conv_txt)
        except Exception as e:
            logger.error(f"Failed to hash {conv_txt}: {e}")
            report.problems.append(Problem(folder=folder_rel, issue="hash_failure"))
            continue

        # Load existing manifest if any
        manifest_path = conv_dir / "manifest.json"
        existing_manifest: Dict[str, Any] = {}
        manifest_issue = None
        if manifest_path.exists():
            try:
                existing_manifest = load_manifest(conv_dir)
                if not existing_manifest:
                    manifest_issue = "manifest_corrupt"
            except Exception as exc:  # pragma: no cover - defensive
                logger.error("Failed to load manifest %s: %s", manifest_path, exc)
                manifest_issue = "manifest_corrupt"

            if manifest_issue:
                report.problems.append(Problem(folder=folder_rel, issue=manifest_issue))

        conv_start, conv_end = _extract_conversation_time_range(conv_txt)
        mtime_iso = _file_mtime_iso(conv_txt)

        # Extract participants from text as fallback/enrichment
        participants = _extract_participants_from_txt(conv_txt)
        last_from, last_to = _extract_last_message_participants(conv_txt)

        # Build new manifest with idempotent timestamps derived from manifest -> convo text -> mtime
        started_at = _preserve_str(
            existing_manifest.get("started_at_utc"), conv_start or mtime_iso
        )
        ended_at = _preserve_str(
            existing_manifest.get("ended_at_utc"), conv_end or started_at
        )

        # Start with a copy to preserve 'messages', 'participants', etc.
        new_manifest = existing_manifest.copy()
        new_manifest.update(
            {
                "manifest_version": "1",
                "folder": folder_rel,
                "subject_label": _preserve_str(
                    existing_manifest.get("subject_label"), folder_rel
                ),
                "participants": participants,  # Explicitly added
                "last_from": last_from,
                "last_to": last_to,
                "message_count": existing_manifest.get("message_count", 0),
                "started_at_utc": started_at,
                "ended_at_utc": ended_at,
                "attachment_count": len(list(attachments_dir.iterdir())),
                "paths": {
                    "conversation_txt": "Conversation.txt",
                    "attachments_dir": "attachments/",
                },
                "sha256_conversation": sha256,
                "conv_id": existing_manifest.get("conv_id"),
                "conv_key_type": existing_manifest.get("conv_key_type"),
            }
        )

        # Check if update needed
        if _manifests_differ(existing_manifest, new_manifest):
            try:
                atomic_write_json(manifest_path, new_manifest)
                if not existing_manifest:
                    report.manifests_created += 1
                    report.problems.append(
                        Problem(folder=folder_rel, issue="manifest_written:created")
                    )
                else:
                    report.manifests_updated += 1
                    report.problems.append(
                        Problem(folder=folder_rel, issue="manifest_written:updated")
                    )
            except Exception as e:
                logger.error(f"Failed to write manifest {manifest_path}: {e}")
                report.problems.append(
                    Problem(folder=folder_rel, issue="manifest_write_failed")
                )

    # Write report
    try:
        report_path = artifacts_dir / "validation_report.json"
        atomic_write_json(report_path, report.model_dump(by_alias=True))
    except Exception as e:
        logger.error(f"Failed to write validation report: {e}")

    return report


def _calculate_conversation_hash(path: Path) -> str:
    """Calculate SHA256 of file with CRLF->LF normalization."""
    sha256 = hashlib.sha256()
    with path.open("rb") as f:
        content = f.read()
        # Normalize line endings to LF
        content = content.replace(b"\r\n", b"\n")
        sha256.update(content)
    return sha256.hexdigest()


def _now_iso() -> str:
    """Current UTC time in ISO format."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _file_mtime_iso(path: Path) -> str:
    """Stable ISO timestamp from file mtime (UTC)."""
    try:
        mtime = path.stat().st_mtime
        return datetime.fromtimestamp(mtime, tz=timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
    except OSError:
        return _now_iso()


def _manifests_differ(old: Dict[str, Any], new: Dict[str, Any]) -> bool:
    """Check if relevant fields differ."""
    # Compare keys that we control/compute
    keys = [
        "manifest_version",
        "folder",
        "sha256_conversation",
        "paths",
        "attachment_count",
        "participants",
        "last_from",
        "last_to",
    ]
    for k in keys:
        if old.get(k) != new.get(k):
            return True
    return False


def _preserve_str(value: Any, fallback: str) -> str:
    """Return value if it is a non-empty string; otherwise fallback."""
    if isinstance(value, str) and value.strip():
        return value
    return fallback


def _extract_conversation_time_range(conv_txt: Path) -> tuple[str | None, str | None]:
    """Extract min/max message timestamps from Conversation.txt content."""
    try:
        text = conv_txt.read_text(encoding="utf-8-sig", errors="replace")
    except Exception:
        return None, None

    fmt_candidates = ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"]
    earliest: datetime | None = None
    latest: datetime | None = None

    for line in text.splitlines():
        # Cheap scan: look for tokens that resemble timestamps (date and time separated by space or T)
        tokens = line.replace("T", " ").split()
        for token in tokens:
            if len(token) < 16:
                continue
            for fmt in fmt_candidates:
                try:
                    if len(token) == 16 and fmt.endswith("%S"):
                        continue
                    dt = datetime.strptime(token, fmt).replace(tzinfo=timezone.utc)
                    if earliest is None or dt < earliest:
                        earliest = dt
                    if latest is None or dt > latest:
                        latest = dt
                    break
                except Exception:
                    continue

    def to_iso(dt: datetime | None) -> str | None:
        if dt is None:
            return None
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    return to_iso(earliest), to_iso(latest)


def _extract_participants_from_txt(conv_txt: Path) -> List[str]:
    """
    Extract participant names/emails from headers in Conversation.txt.

    Args:
        conv_txt: Path to Conversation.txt

    Returns:
        List of distinct participant strings (sorted)
    """
    import re

    participants = set()
    try:
        text = conv_txt.read_text(encoding="utf-8-sig", errors="replace")

        # Regex to find From/To/Cc fields
        # Handles:
        # 1. "From: name <email>" (Start of line)
        # 2. "... | From: email | ..." (Pipe delimited)
        # 3. Avoids matching random "To:" in body text
        clean_pattern = r"(?:^|\|)\s*(?:From|To|Cc):\s*([^|\n\r]+)"

        matches = re.findall(clean_pattern, text, re.IGNORECASE | re.MULTILINE)

        for m in matches:
            clean = m.strip()
            # Heuristics to filter noise:
            # 1. Must be non-empty
            # 2. Should be reasonably short (names/emails usually < 100 chars)
            # 3. Should not look like a sentence (e.g. no "." at end unless it's an initial?)
            if clean and len(clean) < 100:
                # Basic cleanup of surrounding quotes/brackets if regex missed them
                clean = clean.strip(" <>\"'")
                participants.add(clean)

    except Exception as e:
        logger.warning(f"Failed to extract participants from {conv_txt}: {e}")

    # Sort case-insensitively and de-dup
    unique_sorted = sorted(list({p.lower(): p for p in participants}.values()))
    return unique_sorted


def _extract_last_message_participants(
    conv_txt: Path,
) -> tuple[Optional[str], List[str]]:
    """Extract the sender (From) and recipients (To) of the last message in the conversation.

    The Conversation.txt format typically contains lines like:
        2024-10-07 14:43 | From: alice@example.com | To: bob@example.com; carol@example.com
    This function finds the last such line and returns the From address and a list of To addresses.
    """
    import re

    last_from: Optional[str] = None
    last_to: List[str] = []
    try:
        text = conv_txt.read_text(encoding="utf-8-sig", errors="replace")
        # Process lines in reverse to find the last line containing From and To
        for line in reversed(text.splitlines()):
            if "From:" in line and "To:" in line:
                parts = [p.strip() for p in line.split("|")]
                from_part = next((p for p in parts if p.startswith("From:")), None)
                to_part = next((p for p in parts if p.startswith("To:")), None)
                if from_part and to_part:
                    last_from = from_part.split(":", 1)[1].strip()
                    to_raw = to_part.split(":", 1)[1].strip()
                    for addr in re.split(r"[;,]", to_raw):
                        addr = addr.strip()
                        if addr:
                            last_to.append(addr)
                break
    except Exception as e:
        logger.warning(
            f"Failed to extract last message participants from {conv_txt}: {e}"
        )
    return last_from, last_to
