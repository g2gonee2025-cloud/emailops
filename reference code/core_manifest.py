from __future__ import annotations

import json
import logging
import re
from datetime import UTC
from pathlib import Path
from typing import Any

"""
core_manifest.py - Centralized manifest.json parsing and metadata extraction.

This module provides THE SINGLE SOURCE OF TRUTH for:
1. Loading manifest.json files (with robust fallback parsing)
2. Extracting lightweight metadata (for indexing/chunking)
3. Extracting detailed participant information (for summarization)

All other modules should import from here instead of duplicating logic.

NOTE: This module returns plain Python dict/tuple structures for maximum compatibility.
"""

logger = logging.getLogger(__name__)
# Control character pattern
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]")


# ============================================================================
# Manifest File Loading (Canonical with Robust Fallbacks)
# ============================================================================
def load_manifest(convo_dir: Path) -> dict[str, Any]:
    """
    Load and parse manifest.json with multiple fallback strategies.
    THIS IS THE CANONICAL MANIFEST LOADER - use this instead of duplicating.
    Parsing strategies (in order):
    1. Strict JSON (UTF-8 with BOM handling)
    2. Repaired JSON (fixes common backslash issues)
    3. Empty dict (graceful failure)
    Args:
        convo_dir: Path to conversation directory containing manifest.json
    Returns:
        Parsed manifest dict, or {} if loading/parsing fails
    """
    if not isinstance(convo_dir, Path):
        logger.error(
            "load_manifest: Invalid convo_dir type: %s", type(convo_dir).__name__
        )
        return {}
    manifest_path = convo_dir / "manifest.json"
    if not manifest_path.exists():
        logger.debug("load_manifest: Manifest not found at %s", manifest_path)
        return {}
    raw_text = ""
    try:
        raw_bytes = manifest_path.read_bytes()
        try:
            raw_text = raw_bytes.decode("utf-8-sig")
        except UnicodeDecodeError:
            raw_text = raw_bytes.decode("latin-1", errors="ignore")
            logger.warning(
                "Manifest at %s was not valid UTF-8, fell back to latin-1.", convo_dir
            )
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
                logger.error(
                    "Failed to parse manifest for %s: %s. Using empty manifest.",
                    convo_dir,
                    e2,
                )
                return {}
    except Exception as e:
        logger.warning(
            "Unexpected error while loading manifest from %s: %s. Skipping.",
            convo_dir,
            e,
        )
        return {}


# ============================================================================
# Lightweight Metadata Extraction (For Indexing/Chunking)
# ============================================================================
def extract_metadata_lightweight(manifest: dict[str, Any]) -> dict[str, Any]:
    """
    Extract lightweight metadata from parsed manifest for indexing/chunking.
    THIS IS THE CANONICAL METADATA EXTRACTOR.

    Extracts ALL fields needed for search filtering:
    - subject: Smart subject or fallback to subject field
    - from: List of (name, email) pairs aggregated across messages (deduplicated)
    - to: List of (name, email) pairs aggregated across messages (deduplicated)
    - cc: List of (name, email) pairs aggregated across messages (deduplicated)
    - start_date: Start of conversation time span (best-effort)
    - end_date: End of conversation time span (best-effort)

    Args:
        manifest: Parsed manifest dict (from load_manifest or conv["manifest"])

    Returns:
        Dict with keys: subject, from, to, cc, start_date, end_date
    """
    man = manifest or {}
    subject = (man.get("smart_subject") or man.get("subject") or "").strip()

    from_list: list[tuple[str, str]] = []
    to_list: list[tuple[str, str]] = []
    cc_list: list[tuple[str, str]] = []

    seen_from: set[str] = set()
    seen_to: set[str] = set()
    seen_cc: set[str] = set()

    def _add_pair(
        out: list[tuple[str, str]], seen: set[str], name: str, email: str
    ) -> None:
        name_s = (name or "").strip()
        email_s = (email or "").strip()
        if not name_s and not email_s:
            return
        key = email_s.lower() if email_s else f"name:{_normalize_name(name_s)}"
        if not key or key in seen:
            return
        seen.add(key)
        out.append((name_s, email_s))

    # Aggregate from/to/cc across all messages (deduplicated)
    try:
        msgs = man.get("messages") or []
        if isinstance(msgs, list):
            for msg in msgs:
                if not isinstance(msg, dict):
                    continue

                f = msg.get("from")
                if isinstance(f, dict):
                    _add_pair(
                        from_list, seen_from, f.get("name", ""), f.get("smtp", "")
                    )

                for rec in msg.get("to") or []:
                    if isinstance(rec, dict):
                        _add_pair(
                            to_list, seen_to, rec.get("name", ""), rec.get("smtp", "")
                        )

                for rec in msg.get("cc") or []:
                    if isinstance(rec, dict):
                        _add_pair(
                            cc_list, seen_cc, rec.get("name", ""), rec.get("smtp", "")
                        )
    except Exception:
        pass

    # Dates: best-effort min/max across messages (falls back to first/last)
    start_date = None
    end_date = None
    try:
        msgs = man.get("messages") or []
        if isinstance(msgs, list) and msgs:
            raw_start = msgs[0].get("date") if isinstance(msgs[0], dict) else None
            raw_end = msgs[-1].get("date") if isinstance(msgs[-1], dict) else None

            from datetime import datetime

            def _parse_dt(v: Any) -> datetime | None:
                if v is None:
                    return None
                if isinstance(v, (int, float)):
                    try:
                        return datetime.fromtimestamp(float(v), tz=UTC)
                    except Exception:
                        return None
                if isinstance(v, str):
                    s = v.strip()
                    if not s:
                        return None
                    if s.endswith("Z"):
                        s = s[:-1] + "+00:00"
                    try:
                        dt = datetime.fromisoformat(s)
                        if dt.tzinfo is None:
                            dt = dt.replace(tzinfo=UTC)
                        return dt
                    except Exception:
                        return None
                return None

            dts: list[datetime] = []
            for msg in msgs:
                if isinstance(msg, dict) and msg.get("date") is not None:
                    dt = _parse_dt(msg.get("date"))
                    if dt is not None:
                        dts.append(dt)

            if dts:
                start_date = min(dts).astimezone(UTC).isoformat()
                end_date = max(dts).astimezone(UTC).isoformat()
            else:
                start_date = raw_start
                end_date = raw_end
    except Exception:
        pass

    return {
        "subject": subject,
        "from": from_list,
        "to": to_list,
        "cc": cc_list,
        "start_date": start_date,
        "end_date": end_date,
    }


# ============================================================================
# Detailed Participant Extraction (For Summarization)
# ============================================================================
def extract_participants_detailed(
    manifest: dict[str, Any],
    default_role: str = "other",
    default_tone: str = "neutral",
    default_stance: str = "N/A",
    max_participants: int = 25,
) -> list[dict[str, str]]:
    """
    Extract detailed participant information from manifest for summarization.

    Returns rich participant schema with:
      - name: Participant name
      - email: Email address
      - role: One of [client, broker, underwriter, internal, other]
      - tone: One of [professional, frustrated, urgent, friendly, demanding, neutral]
      - stance: Their position/attitude in the thread

    Notes:
      - Aggregates participants across all messages (sender + recipients).
      - Deduplicates by email (preferred) or normalized name.
      - Applies defaults; downstream summarization can refine role/tone/stance.

    Args:
        manifest: Parsed manifest dict
        default_role: Default role when unknown (default: "other")
        default_tone: Default tone when unknown (default: "neutral")
        default_stance: Default stance when unknown (default: "N/A")
        max_participants: Maximum number to return (default: 25)

    Returns:
        List of participant dicts with full schema, deduplicated
    """
    out: list[dict[str, str]] = []
    if not isinstance(manifest, dict):
        logger.warning(
            "extract_participants_detailed: Non-dict manifest type: %s",
            type(manifest).__name__,
        )
        return []

    def _mk(name: str, email: str, role: str = "other") -> dict[str, str]:
        return {
            "name": _safe_str(name, 80),
            "role": role,
            "email": _safe_str(email, 120),
            "tone": default_tone,
            "stance": default_stance,
        }

    try:
        messages = manifest.get("messages")
        if not isinstance(messages, list) or not messages:
            logger.debug("extract_participants_detailed: No messages in manifest")
            return []

        for msg in messages:
            if not isinstance(msg, dict):
                continue

            # Sender
            f = msg.get("from")
            if isinstance(f, dict):
                out.append(_mk(f.get("name", ""), f.get("smtp", ""), role=default_role))

            # Recipients
            for rec in msg.get("to") or []:
                if isinstance(rec, dict):
                    out.append(
                        _mk(rec.get("name", ""), rec.get("smtp", ""), role=default_role)
                    )

            for rec in msg.get("cc") or []:
                if isinstance(rec, dict):
                    out.append(
                        _mk(rec.get("name", ""), rec.get("smtp", ""), role=default_role)
                    )

    except (TypeError, AttributeError, KeyError) as e:
        logger.warning(
            "extract_participants_detailed: Error extracting participants: %s", e
        )
    except Exception as e:
        logger.error("extract_participants_detailed: Unexpected error: %s", e)

    # Deduplicate by lowercase email or normalized name; skip empty entries
    seen: set[str] = set()
    deduped: list[dict[str, str]] = []
    for p_item in out:
        email_key = (p_item.get("email") or "").lower()
        name_key = _normalize_name(p_item.get("name", ""))
        if not email_key and not name_key:
            continue
        key = email_key or f"name:{name_key}"
        if key in seen:
            continue
        seen.add(key)
        deduped.append(p_item)
        if len(deduped) >= max_participants:
            break

    return deduped


# ============================================================================
# Helper Functions
# ============================================================================
def _safe_str(v: Any, max_len: int) -> str:
    """
    Safely convert any value to string with length limit.
    Handles None values and enforces maximum length.
    """
    if max_len < 0:
        logger.warning("_safe_str: Negative max_len %d, using 0", max_len)
        max_len = 0
    try:
        s = "" if v is None else str(v)
        return s[:max_len].rstrip() if len(s) > max_len else s
    except Exception as e:
        logger.error("_safe_str: Failed to convert value: %s", e)
        return ""


def _normalize_name(n: str) -> str:
    """
    Normalize a name for de-duplication (simple lower/strip).
    Handles non-string inputs gracefully.
    """
    if not n or not isinstance(n, str):
        return ""
    try:
        return re.sub(r"\s+", " ", n.strip().lower())
    except Exception as e:
        logger.warning("_normalize_name: Failed to normalize '%s': %s", n, e)
        return ""


# ============================================================================
# Convenience Function (Combines Loading + Extraction)
# ============================================================================
def get_conversation_metadata(convo_dir: Path) -> dict[str, Any]:
    """
    ONE-STOP function: Load manifest.json and extract lightweight metadata.
    Combines load_manifest() + extract_metadata_lightweight() for convenience.
    Use this when you need metadata and don't already have the manifest loaded.
    Args:
        convo_dir: Path to conversation directory
    Returns:
        Dict with keys: subject, participants, start_date, end_date
    """
    manifest = load_manifest(convo_dir)
    return extract_metadata_lightweight(manifest)


__all__ = [
    "extract_metadata_lightweight",
    "extract_participants_detailed",
    "get_conversation_metadata",
    "load_manifest",
]
