"""
Conversation loading utilities.
Handles loading conversation content, manifest/summary JSON, and attachment texts.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from cortex.config.loader import get_config
from cortex.ingestion.core_manifest import load_manifest
from cortex.text_extraction import extract_text
from cortex.utils import read_text_file, scrub_json

logger = logging.getLogger(__name__)

# Control character pattern
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]")


def _load_conversation_text(convo_dir: Path) -> str:
    """Load and sanitize the main conversation text file."""
    candidate_names = ["Conversation.txt", "conversation.txt"]
    for name in candidate_names:
        convo_txt_path = convo_dir / name
        if not convo_txt_path.exists():
            continue
        try:
            if name != "Conversation.txt":
                logger.debug(
                    "Using alternate conversation file name '%s' in %s", name, convo_dir
                )
            return read_text_file(convo_txt_path)
        except OSError as e:
            logger.warning("Failed to read %s at %s: %s", name, convo_dir, e)
            return ""
    return ""


def _load_summary(convo_dir: Path) -> dict[str, Any]:
    """Load and parse the summary.json file."""
    summary_json_path = convo_dir / "summary.json"
    if not summary_json_path.exists():
        return {}
    try:
        return scrub_json(json.loads(summary_json_path.read_text(encoding="utf-8-sig")))
    except (json.JSONDecodeError, OSError):
        return {}


def _process_single_attachment(
    att_file: Path,
    max_attachment_text_chars: int,
    skip_if_attachment_over_mb: float,
) -> dict[str, Any] | None:
    """Helper to process a single attachment file."""
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
                    return None
            except OSError:
                pass

        txt = extract_text(att_file, max_chars=max_attachment_text_chars)
        if txt.strip():
            return {"path": str(att_file), "text": txt}
    except OSError as e:
        logger.warning("Failed to process attachment %s: %s", att_file, e)
    return None


def _process_attachments(
    convo_dir: Path,
    include_attachment_text: bool,
    max_total_attachment_text: int,
    max_attachment_text_chars: int,
    skip_if_attachment_over_mb: float,
) -> tuple[list[dict[str, Any]], str]:
    """Process all attachments for a conversation."""
    attachments = []
    appended_text = ""
    total_appended = 0
    attachment_files = _collect_attachment_files(convo_dir)

    for att_file in attachment_files:
        attachment_data = _process_single_attachment(
            att_file,
            max_attachment_text_chars,
            skip_if_attachment_over_mb,
        )
        if not attachment_data:
            continue

        attachments.append(attachment_data)

        if include_attachment_text and total_appended < max_total_attachment_text:
            remaining = max_total_attachment_text - total_appended
            try:
                rel_path = att_file.relative_to(convo_dir)
            except ValueError:
                rel_path = att_file.name
            header = f"\n\n--- ATTACHMENT: {rel_path} ---\n\n"
            snippet = attachment_data["text"][: max(0, remaining - len(header))]
            appended_text += header + snippet
            total_appended += len(header) + len(snippet)

    return attachments, appended_text


def load_conversation(
    convo_dir: Path,
    *,
    include_attachment_text: bool = False,
    max_total_attachment_text: int | None = None,
    max_attachment_text_chars: int | None = None,
    skip_if_attachment_over_mb: float | None = None,
) -> dict[str, Any] | None:
    """
    Load conversation content, manifest/summary JSON, and attachment texts.

    All parameters except convo_dir are keyword-only to prevent argument confusion.

    Args:
        convo_dir: Path to conversation directory containing Conversation.txt and manifest.json
        include_attachment_text: Whether to include attachment text in conversation_txt
        max_total_attachment_text: Max total chars from all attachments (uses config default if None)
        max_attachment_text_chars: Max chars per attachment (uses config default if None)
        skip_if_attachment_over_mb: Skip attachments larger than this (uses config default if None)

    Returns:
        dict with keys: path, conversation_txt, attachments, summary, manifest
        Returns None if conversation directory is invalid or contains no processable content

    Raises:
        OSError: If directory cannot be accessed
        ValueError: If convo_dir is not a valid Path
    """
    # Validate input path
    if not convo_dir or not isinstance(convo_dir, Path):
        logger.error("Invalid conversation directory path: %s", convo_dir)
        return None

    if not convo_dir.exists():
        logger.error("Conversation directory does not exist: %s", convo_dir)
        return None

    if not convo_dir.is_dir():
        logger.error("Path is not a directory: %s", convo_dir)
        return None

    config = get_config().limits
    max_total_attachment_text = (
        max_total_attachment_text
        if max_total_attachment_text is not None
        else int(
            config.max_total_attachments_mb * 1024 * 1024
        )  # Approximate char count from MB
    )
    max_attachment_text_chars = (
        max_attachment_text_chars
        if max_attachment_text_chars is not None
        else config.max_attachment_text_chars
    )
    skip_if_attachment_over_mb = (
        skip_if_attachment_over_mb
        if skip_if_attachment_over_mb is not None
        else config.skip_attachment_over_mb
    )

    conversation_text = _load_conversation_text(convo_dir)
    attachments, appended_text = _process_attachments(
        convo_dir,
        include_attachment_text,
        max_total_attachment_text,
        max_attachment_text_chars,
        skip_if_attachment_over_mb,
    )
    conversation_text += appended_text

    manifest = {}
    if (convo_dir / "manifest.json").exists():
        manifest = scrub_json(load_manifest(convo_dir))

    # Validate that we have some minimal content
    if not conversation_text and not attachments and not manifest:
        logger.warning(
            "Conversation directory contains no processable content: %s", convo_dir
        )
        return None

    return {
        "path": str(convo_dir),
        "conversation_txt": conversation_text,
        "attachments": attachments,
        "summary": _load_summary(convo_dir),
        "manifest": manifest,
    }


def _collect_attachment_files(convo_dir: Path) -> list[Path]:
    """
    Collect attachment files, avoiding duplicates and sorting deterministically.
    Refactored for performance and reduced I/O.
    """
    all_files: set[Path] = set()
    excluded = {"Conversation.txt", "manifest.json", "summary.json"}

    # Pass 1: Collect files from conversation directory
    try:
        for child in convo_dir.iterdir():
            if child.is_file() and child.name not in excluded:
                all_files.add(child)
    except (OSError, PermissionError) as e:
        logger.warning("Failed to iterate conversation dir %s: %s", convo_dir, e)

    # Pass 2: Collect files from Attachments subdirectory
    attachments_dir = convo_dir / "Attachments"
    if attachments_dir.is_dir():
        try:
            for p in attachments_dir.rglob("*"):
                if p.is_file():
                    all_files.add(p)
        except (OSError, PermissionError) as e:
            logger.warning(
                "Failed to iterate Attachments directory at %s: %s", attachments_dir, e
            )

    # Sort deterministically
    sorted_files = sorted(
        all_files, key=lambda p: (p.parent.as_posix(), p.name.lower())
    )

    return sorted_files
