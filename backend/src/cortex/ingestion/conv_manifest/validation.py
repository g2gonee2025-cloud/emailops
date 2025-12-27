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
import json

from cortex.ingestion.core_manifest import extract_metadata_lightweight, load_manifest
from cortex.ingestion.models import Problem
from cortex.utils.atomic_io import atomic_write_json
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class SecurityException(Exception):
    """Custom exception for security-related errors."""


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
    from cortex.config.loader import get_config

    config = get_config()
    try:
        export_root = config.directories.export_root.resolve(strict=True)
        scan_root = root.resolve(strict=True)
        if not scan_root.is_relative_to(export_root):
            raise SecurityException("Path traversal attempt")

    except (SecurityException, FileNotFoundError) as e:
        logger.error("Path validation failed for provided root path: %s", e)
        report = ManifestValidationReport(root=str(root))
        report.problems.append(Problem(folder=str(root), issue="path_validation_failed"))
        return report

    report = ManifestValidationReport(root=str(root.absolute()))

    if not root.exists():
        logger.error("Export root not found: %s", root)
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
        except IOError as e:
            logger.error("Failed to hash %s: %s", conv_txt.relative_to(root), e)
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
            except (IOError, json.JSONDecodeError) as exc:  # pragma: no cover - defensive
                logger.error("Failed to load manifest %s: %s", manifest_path, exc)
                manifest_issue = "manifest_corrupt"

            if manifest_issue:
                report.problems.append(Problem(folder=folder_rel, issue=manifest_issue))

        # Use the canonical metadata extractor
        metadata = extract_metadata_lightweight(existing_manifest)
        mtime_iso = _file_mtime_iso(conv_txt)

        # Build new manifest with idempotent timestamps derived from manifest -> convo text -> mtime
        started_at = _preserve_str(
            existing_manifest.get("started_at_utc"), metadata.get("start_date") or mtime_iso
        )
        ended_at = _preserve_str(
            existing_manifest.get("ended_at_utc"), metadata.get("end_date") or started_at
        )

        # Start with a copy to preserve 'messages', 'participants', etc.
        new_manifest = existing_manifest.copy()
        all_participants = metadata.get("from", []) + metadata.get("to", []) + metadata.get("cc", [])
        # Extract email (index 1 of tuple), filter out empty ones, ensure uniqueness, and sort
        participant_emails = sorted(
            list(set(p[1] for p in all_participants if p and p[1])),
            key=str.lower
        )
        new_manifest.update(
            {
                "manifest_version": "1",
                "folder": folder_rel,
                "subject_label": _preserve_str(
                    existing_manifest.get("subject_label"), metadata.get("subject") or folder_rel
                ),
                "participants": participant_emails,
                "last_from": (metadata.get("from")[-1] if metadata.get("from") else [("", "")])[-1][1],
                "last_to": [p[1] for p in metadata.get("to", [])],
                "message_count": existing_manifest.get("message_count", 0),
                "started_at_utc": started_at,
                "ended_at_utc": ended_at,
                "attachment_count": sum(1 for _ in attachments_dir.iterdir()),
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
            except IOError as e:
                logger.error(
                    "Failed to write manifest %s: %s",
                    manifest_path.relative_to(root),
                    e,
                )
                report.problems.append(
                    Problem(folder=folder_rel, issue="manifest_write_failed")
                )

    # Write report
    try:
        report_path = artifacts_dir / "validation_report.json"
        atomic_write_json(report_path, report.model_dump(by_alias=True))
    except IOError as e:
        logger.error("Failed to write validation report: %s", e)

    return report


def _calculate_conversation_hash(path: Path) -> str:
    """Calculate SHA256 of file with CRLF->LF normalization, reading in chunks."""
    sha256 = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(65536):  # 64k chunks
            sha256.update(chunk.replace(b"\r\n", b"\n"))
    return sha256.hexdigest()


def _now_iso() -> str:
    """Current UTC time in ISO format."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _file_mtime_iso(path: Path) -> str:
    """Stable ISO timestamp from file mtime (UTC)."""
    mtime = path.stat().st_mtime
    return datetime.fromtimestamp(mtime, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


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
