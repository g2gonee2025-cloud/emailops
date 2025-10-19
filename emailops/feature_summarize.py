#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import csv
import inspect
import json
import logging
import os
import random
import re
import tempfile
import time

# Python 3.10 compatibility for UTC
from datetime import UTC, datetime
from io import StringIO
from pathlib import Path
from typing import Any

# Import strategy: Try package imports first, fail fast with clear error
try:
    from emailops.llm_client_shim import complete_json, complete_text
    from emailops.util_main import (
        clean_email_text,
        ensure_dir,
        extract_email_metadata,
        logger,
        read_text_file,
    )
except ImportError as e:
    raise ImportError(
        "Failed to import required emailops modules. "
        "Please ensure the emailops package is properly installed or "
        "that all required modules are in your PYTHONPATH.\n"
        f"Original error: {e}"
    ) from e

# =============================================================================
# Configuration (env-overridable) to keep prompts bounded and predictable
# =============================================================================
ANALYZER_VERSION = os.getenv("SUMMARIZER_VERSION", "2.2-facts-ledger")

MAX_THREAD_CHARS = int(os.getenv("SUMMARIZER_THREAD_MAX_CHARS", "16000"))
CRITIC_THREAD_CHARS = int(os.getenv("SUMMARIZER_CRITIC_MAX_CHARS", "5000"))
IMPROVE_THREAD_CHARS = int(os.getenv("SUMMARIZER_IMPROVE_MAX_CHARS", "8000"))

# Hard caps to avoid pathological JSON explosions
MAX_PARTICIPANTS = int(os.getenv("SUMMARIZER_MAX_PARTICIPANTS", "25"))
MAX_SUMMARY_POINTS = int(os.getenv("SUMMARIZER_MAX_SUMMARY_POINTS", "25"))
MAX_NEXT_ACTIONS = int(os.getenv("SUMMARIZER_MAX_NEXT_ACTIONS", "50"))
MAX_FACT_ITEMS = int(os.getenv("SUMMARIZER_MAX_FACT_ITEMS", "50"))
SUBJECT_MAX_LEN = int(os.getenv("SUMMARIZER_SUBJECT_MAX_LEN", "100"))

DEFAULT_CATALOG = [
    "insurance_coverage_query",
    "contract_review_request",
    "certificate_request",
    "endorsement_change_request",
    "claim_notification_or_management",
    "claim_update_request",
    "admin_internal",
    "admin_external",
    "other",
]


# =============================================================================
# Internal helpers
# =============================================================================
def _extract_first_balanced_json_object(s: str) -> str | None:
    """
    Finds the first balanced JSON object (from '{' to '}') in a string.
    Properly handles nested braces, strings with braces, and escaped characters.
    HIGH #15: Enhanced to correctly handle braces inside string values.
    Returns the object as a string, or None if no balanced object is found.
    """

    # Type and empty validation
    if not isinstance(s, str):
        logger.warning(
            "_extract_first_balanced_json_object: Non-string input type: %s",
            type(s).__name__,
        )
        return None

    if not s or "{" not in s:
        logger.debug("_extract_first_balanced_json_object: No JSON object found in string")
        return None

    first_brace = s.find("{")
    if first_brace == -1:
        return None

    balance = 0
    in_string = False
    escape_next = False

    try:
        for i, char in enumerate(s[first_brace:]):
            # HIGH #15: Improved escape handling
            if escape_next:
                # Character is escaped, don't process it
                escape_next = False
                continue

            if char == "\\" and in_string:
                # Next character will be escaped
                escape_next = True
                continue

            # HIGH #15: Only toggle string state on unescaped quotes
            if char == '"' and not escape_next:
                in_string = not in_string
                continue

            # HIGH #15: Only count braces when NOT inside a string
            if not in_string:
                if char == "{":
                    balance += 1
                elif char == "}":
                    balance -= 1

                if balance == 0 and i > 0:  # Ensure we've closed the first brace
                    result = s[first_brace : first_brace + i + 1]
                    logger.debug(
                        "_extract_first_balanced_json_object: Extracted %d char JSON object",
                        len(result),
                    )
                    return result
    except Exception as e:
        logger.error("_extract_first_balanced_json_object: Unexpected error: %s", e)
        return None

    logger.debug("_extract_first_balanced_json_object: No balanced object found")
    return None


def _try_load_json(data: Any) -> dict[str, Any]:
    """
    Robustly parse JSON from model output, accepting dict, bytes, or string.
    Handles:
      1) Pre-parsed dicts
      2) Byte strings (UTF-8 decoded)
      3) Direct JSON strings
      4) Lists containing dicts (scans for valid analysis dict)
      5) Fenced ```json blocks
      6) First balanced {...} object in the string
    Raises ValueError if not recoverable.
    """
    if isinstance(data, dict):
        logger.debug("JSON parsing: Already a dict, returning as-is")
        return data
    if data is None or (isinstance(data, (str, bytes)) and not data):
        logger.warning("JSON parsing: Empty or None data provided")
        raise ValueError("Empty or None data provided for JSON parsing")

    s = ""
    if isinstance(data, bytes):
        try:
            s = data.decode("utf-8")
            logger.debug("JSON parsing: Decoded %d bytes from UTF-8", len(data))
        except UnicodeDecodeError as e:
            logger.error("JSON parsing: Failed to decode bytes as UTF-8: %s", e)
            raise ValueError(f"Failed to decode bytes as UTF-8: {e}") from e
    elif isinstance(data, str):
        s = data
        logger.debug("JSON parsing: Processing string of %d chars", len(s))
    else:
        logger.error("JSON parsing: Unsupported data type: %s", type(data).__name__)
        raise ValueError(f"Unsupported data type: {type(data).__name__}") from None

    s = s.strip()
    if not s:
        logger.warning("JSON parsing: String is empty after stripping")
        raise ValueError("Empty string after stripping whitespace") from None

    # 1) Try direct parsing first
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            logger.debug("JSON parsing: Direct parsing successful")
            return obj
        # Handle list containing potential analysis dicts
        if isinstance(obj, list):
            logger.debug("JSON parsing: Direct parse gave list, scanning for valid dict")
            known_keys = {"category", "participants", "facts_ledger", "summary", "next_actions", "risk_indicators"}
            for item in obj:
                if isinstance(item, dict):
                    # Check if dict has at least 2 known top-level keys
                    matching_keys = sum(1 for k in known_keys if k in item)
                    if matching_keys >= 2:
                        logger.debug("JSON parsing: Found valid dict in list with %d matching keys", matching_keys)
                        return item
        logger.warning("JSON parsing: Direct parse gave non-dict: %s", type(obj).__name__)
    except json.JSONDecodeError as e:
        logger.debug("JSON parsing: Direct parse failed: %s", e)

    # 2) Look for fenced ```json blocks
    fenced_matches = list(re.finditer(r"```(?:json)?\s*([\s\S]*?)\s*```", s, flags=re.IGNORECASE))
    logger.debug("JSON parsing: Found %d fenced code blocks", len(fenced_matches))

    for i, m in enumerate(fenced_matches):
        block = _extract_first_balanced_json_object(m.group(1))
        if block:
            try:
                obj = json.loads(block)
                if isinstance(obj, dict):
                    logger.debug("JSON parsing: Extracted from fenced block #%d", i + 1)
                    return obj
            except json.JSONDecodeError as e:
                logger.debug("JSON parsing: Failed to parse fenced block #%d: %s", i + 1, e)

    # 3) Fallback to the first balanced object in the whole string
    block = _extract_first_balanced_json_object(s)
    if block:
        try:
            obj = json.loads(block)
            if isinstance(obj, dict):
                logger.debug("JSON parsing: Extracted balanced object from raw string")
                return obj
        except json.JSONDecodeError as e:
            logger.debug("JSON parsing: Failed to parse balanced object: %s", e)

    # All strategies failed
    logger.error("JSON parsing: All parsing strategies failed for %d char string", len(s))
    sample = s[:200] + "..." if len(s) > 200 else s
    raise ValueError(f"Failed to parse JSON from any strategy. Sample: {sample}")


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


def _coerce_enum(val: Any, allowed: set[str], default: str, synonyms: dict[str, str] | None = None) -> str:
    """
    Coerce a value to an allowed enum with synonym mapping.

    Args:
        val: The value to coerce
        allowed: Set of allowed enum values
        default: Default value if coercion fails
        synonyms: Optional mapping of synonyms to canonical values

    Returns:
        Coerced enum value or default
    """
    if val is None or not isinstance(val, str):
        return default

    # Normalize: lowercase, trim, convert - to _
    normalized = val.strip().lower().replace("-", "_")

    # Apply synonyms if provided
    if synonyms:
        normalized = synonyms.get(normalized, normalized)

    # Check if in allowed set
    if normalized in allowed:
        return normalized

    # Fallback to default
    return default


def _md_escape(v: Any) -> str:
    """
    Escape text for safe Markdown rendering in generated summaries.
    Handles None values and conversion errors gracefully.
    """
    if v is None:
        return ""

    try:
        s = str(v)
        # Escape common markdown-sensitive characters
        return re.sub(r"([\\`*_{}\[\]()#+\-.!>])", r"\\\1", s)
    except Exception as e:
        logger.warning("_md_escape: Failed to escape value: %s", e)
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


def _normalize_subject_line(s: str) -> str:
    """
    Clean up common subject prefixes and whitespace.
    - Collapse repeated 'Re:' / 'Fwd:' chains to a single prefix.
    - Normalize spacing after the prefix to 'Re: <subject>' or 'Fwd: <subject>'.
    """
    if not isinstance(s, str):
        logger.debug("_normalize_subject_line: Non-string input type: %s", type(s).__name__)
        return ""

    try:
        subj = s.strip()
        if not subj:
            return ""

        # Collapse prefixes like "Re: Fwd: Re: " down to a single one.
        # The logic is to find the last prefix (case-insensitive) and take it and everything after.
        # Regex: find the last occurrence of `re:`, `fwd:`, or `fw:` with optional whitespace and colons.
        last_prefix_match = None
        for match in re.finditer(r"(re|fw(d)?):\s*", subj, re.IGNORECASE):
            last_prefix_match = match

        if last_prefix_match:
            prefix_end = last_prefix_match.end()
            prefix_type = last_prefix_match.group(1).lower()
            # Keep "Fwd:" if present, otherwise default to "Re:"
            final_prefix = "Fwd:" if "fw" in prefix_type else "Re:"
            return f"{final_prefix} {subj[prefix_end:].strip()}".strip()

        return subj  # Return original if no prefixes are found

    except Exception as e:
        logger.error("_normalize_subject_line: Failed to normalize '%s': %s", s[:50], e)
        return s.strip() if isinstance(s, str) else ""


def _normalize_analysis(data: Any, catalog: list[str]) -> dict[str, Any]:
    """
    Coerce potentially imperfect LLM output to the required schema
    with safe defaults and type checks.
    Handles None, invalid types, and malformed data gracefully.
    """
    # Input validation
    if data is None:
        logger.warning("_normalize_analysis: None data provided, using empty dict")
        d: dict[str, Any] = {}
    elif isinstance(data, dict):
        d = dict(data)
    else:
        logger.warning(
            "_normalize_analysis: Non-dict data type %s, using empty dict",
            type(data).__name__,
        )
        d = {}

    # Validate catalog - ensure it always has a catch-all
    if not catalog or not isinstance(catalog, list):
        logger.warning("_normalize_analysis: Invalid catalog, using DEFAULT_CATALOG")
        catalog = DEFAULT_CATALOG
    else:
        # Ensure catalog has a catch-all
        if "other" not in catalog:
            catalog = [*list(catalog), "other"]

    # Required top-level keys
    d.setdefault("category", catalog[-1])
    d.setdefault("subject", "Email thread")
    d.setdefault("participants", [])
    d.setdefault("facts_ledger", {})
    d.setdefault("summary", [])
    d.setdefault("next_actions", [])
    d.setdefault("risk_indicators", [])

    # Validate/normalize category
    if d.get("category") not in catalog:
        d["category"] = catalog[-1]

    # Always normalize and cap subject
    subj = d.get("subject", "Email thread")
    if not isinstance(subj, str):
        subj = "Email thread"
    normalized_subj = _normalize_subject_line(subj)
    d["subject"] = _safe_str(normalized_subj, SUBJECT_MAX_LEN)

    # Types
    if not isinstance(d["participants"], list):
        d["participants"] = []
    else:
        # Normalize each participant object to satisfy schema expectations
        allowed_roles = {"client", "broker", "underwriter", "internal", "other"}
        allowed_tones = {
            "professional",
            "frustrated",
            "urgent",
            "friendly",
            "demanding",
            "neutral",
        }
        parts: list[dict[str, str]] = []
        seen_keys: set = set()
        for p in d["participants"]:
            if not isinstance(p, dict):
                continue
            name = _safe_str(p.get("name", ""), 80)
            email = _safe_str(p.get("email", ""), 120)
            role = p.get("role") if p.get("role") in allowed_roles else "other"
            tone = p.get("tone") if p.get("tone") in allowed_tones else "neutral"
            stance = _safe_str(p.get("stance", "N/A"), 200)
            # Drop entries with neither name nor email
            if not name and not email:
                continue
            norm = {
                "name": name,
                "role": role,
                "email": email,
                "tone": tone,
                "stance": stance,
            }
            # De-dupe by email or normalized name
            email_key = email.lower()
            name_key = _normalize_name(name)
            key = email_key or f"name:{name_key}"
            if key and key in seen_keys:
                continue
            if key:
                seen_keys.add(key)
            parts.append(norm)
        d["participants"] = parts

    if not isinstance(d["summary"], list):
        d["summary"] = [str(d.get("summary", ""))] if d.get("summary") else []
    if not isinstance(d["next_actions"], list):
        d["next_actions"] = []
    if not isinstance(d["risk_indicators"], list):
        d["risk_indicators"] = []

    # Facts ledger shape
    fl = d.get("facts_ledger")
    if not isinstance(fl, dict):
        fl = {}
    fl.setdefault("known_facts", [])
    fl.setdefault("key_dates", [])
    fl.setdefault("commitments_made", [])
    fl.setdefault("required_for_resolution", [])
    fl.setdefault("what_we_have", [])
    fl.setdefault("what_we_need", [])
    fl.setdefault("materiality_for_company", [])
    fl.setdefault("materiality_for_me", [])

    # Ensure lists and coerce nested enums
    for k in (
        "known_facts",
        "key_dates",
        "commitments_made",
        "required_for_resolution",
        "what_we_have",
        "what_we_need",
        "materiality_for_company",
        "materiality_for_me",
    ):
        if not isinstance(fl.get(k), list):
            fl[k] = []

    # Define enum synonyms

    status_synonyms = {
        "pending": "pending",
        "acknowledged": "acknowledged",
        "in_progress": "in_progress",
        "in-progress": "in_progress",
        "inprogress": "in_progress",
        "completed": "completed",
        "done": "completed",
        "blocked": "blocked",
    }

    feasibility_synonyms = {
        "achievable": "achievable",
        "challenging": "challenging",
        "risky": "risky",
        "impossible": "impossible",
        "easy": "achievable",
        "hard": "challenging",
        "difficult": "challenging",
    }

    importance_synonyms = {
        "critical": "critical",
        "important": "important",
        "reference": "reference",
        "info": "reference",
        "informational": "reference",
    }

    priority_synonyms = {
        "critical": "critical",
        "high": "high",
        "med": "medium",
        "medium": "medium",
        "low": "low",
        "normal": "medium",
    }

    # No enums in the new facts_ledger structure, so this section is removed.

    # Coerce enums in commitments_made
    coerced_commits = []
    for commit in fl.get("commitments_made", []):
        if isinstance(commit, dict):
            commit["feasibility"] = _coerce_enum(
                commit.get("feasibility"),
                {"achievable", "challenging", "risky", "impossible"},
                "achievable",
                feasibility_synonyms,
            )
            coerced_commits.append(commit)
    fl["commitments_made"] = coerced_commits

    # Coerce enums in key_dates
    coerced_dates = []
    for kd in fl.get("key_dates", []):
        if isinstance(kd, dict):
            kd["importance"] = _coerce_enum(
                kd.get("importance"), {"critical", "important", "reference"}, "reference", importance_synonyms
            )
            coerced_dates.append(kd)
    fl["key_dates"] = coerced_dates

    d["facts_ledger"] = fl

    # Coerce enums in next_actions
    coerced_actions = []
    for action in d.get("next_actions", []):
        if isinstance(action, dict):
            action["status"] = _coerce_enum(
                action.get("status"), {"open", "in_progress", "blocked", "completed"}, "open", status_synonyms
            )
            action["priority"] = _coerce_enum(
                action.get("priority"), {"critical", "high", "medium", "low"}, "medium", priority_synonyms
            )
            coerced_actions.append(action)
    d["next_actions"] = coerced_actions

    # Apply size caps defensively using direct slicing
    d["participants"] = d["participants"][:MAX_PARTICIPANTS] if isinstance(d["participants"], list) else []
    d["summary"] = d["summary"][:MAX_SUMMARY_POINTS] if isinstance(d["summary"], list) else []
    d["next_actions"] = d["next_actions"][:MAX_NEXT_ACTIONS] if isinstance(d["next_actions"], list) else []

    for k in (
        "known_facts",
        "key_dates",
        "commitments_made",
        "required_for_resolution",
        "what_we_have",
        "what_we_need",
        "materiality_for_company",
        "materiality_for_me",
    ):
        fl[k] = fl.get(k, [])[:MAX_FACT_ITEMS] if isinstance(fl.get(k), list) else []
    d["facts_ledger"] = fl

    return d


def _read_manifest(convo_dir: Path) -> dict[str, Any]:
    """
    Lightweight manifest reader with BOM tolerance and basic sanitation.
    Returns {} if unavailable or invalid.
    """
    if not isinstance(convo_dir, Path):
        logger.error("_read_manifest: Invalid convo_dir type: %s", type(convo_dir).__name__)
        return {}

    manifest_path = convo_dir / "manifest.json"

    if not manifest_path.exists():
        logger.debug("_read_manifest: Manifest not found at %s", manifest_path)
        return {}

    try:
        text = manifest_path.read_text(encoding="utf-8-sig")

        if not text.strip():
            logger.warning("_read_manifest: Empty manifest file at %s", manifest_path)
            return {}

        # Minimal sanitation: drop control chars that break JSON
        text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]", "", text)

        parsed = json.loads(text)

        if not isinstance(parsed, dict):
            logger.warning("_read_manifest: Manifest is not a dict at %s", manifest_path)
            return {}

        logger.debug("_read_manifest: Successfully loaded manifest from %s", manifest_path)
        return parsed

    except json.JSONDecodeError as e:
        logger.warning("_read_manifest: JSON decode error at %s: %s", manifest_path, e)
        return {}
    except OSError as e:
        logger.warning("_read_manifest: File read error at %s: %s", manifest_path, e)
        return {}
    except Exception as e:
        logger.error("_read_manifest: Unexpected error at %s: %s", manifest_path, e)
        return {}


def _participants_from_manifest(manifest: dict[str, Any]) -> list[dict[str, str]]:
    """
    Convert first-message participants in manifest to summarizer schema.
    Roles default to 'other'
    tone 'neutral'
    stance 'N/A' to avoid assumptions.
    """
    out: list[dict[str, str]] = []

    if not isinstance(manifest, dict):
        logger.warning(
            "_participants_from_manifest: Non-dict manifest type: %s",
            type(manifest).__name__,
        )
        return []

    try:
        messages = manifest.get("messages")
        if not isinstance(messages, list) or not messages:
            logger.debug("_participants_from_manifest: No messages in manifest")
            return []

        first = messages[0]
        if not isinstance(first, dict):
            logger.warning("_participants_from_manifest: First message is not a dict")
            return []

        def _mk(name: str, email: str, role: str = "other") -> dict[str, str]:
            return {
                "name": _safe_str(name, 80),
                "role": role,
                "email": _safe_str(email, 120),
                "tone": "neutral",
                "stance": "N/A",
            }

        # Extract 'from' participant
        if first.get("from") and isinstance(first["from"], dict):
            f = first["from"]
            out.append(_mk(f.get("name", ""), f.get("smtp", ""), role="other"))

        # Extract 'to' participants
        to_list = first.get("to")
        if isinstance(to_list, list):
            for rec in to_list:
                if isinstance(rec, dict):
                    out.append(_mk(rec.get("name", ""), rec.get("smtp", ""), role="other"))

        # Extract 'cc' participants
        cc_list = first.get("cc")
        if isinstance(cc_list, list):
            for rec in cc_list:
                if isinstance(rec, dict):
                    out.append(_mk(rec.get("name", ""), rec.get("smtp", ""), role="other"))

    except (TypeError, AttributeError, KeyError) as e:
        logger.warning("_participants_from_manifest: Error extracting participants: %s", e)
        return out
    except Exception as e:
        logger.error("_participants_from_manifest: Unexpected error: %s", e)
        return out
    # de-duplicate by lowercase email or normalized name; skip empty entries
    seen: set = set()
    deduped: list[dict[str, str]] = []
    for p in out:
        email_key = (p.get("email") or "").lower()
        name_key = _normalize_name(p.get("name", ""))
        if not email_key and not name_key:
            continue
        key = email_key or f"name:{name_key}"
        if key in seen:
            continue
        seen.add(key)
        deduped.append(p)
    return deduped[:MAX_PARTICIPANTS] if deduped else []


def _merge_manifest_into_analysis(analysis: dict[str, Any], convo_dir: Path, raw_thread_text: str) -> dict[str, Any]:
    """
    Merge subject/participants/dates from manifest + raw headers when the model
    couldn't infer them reliably. Never overrides non-empty, already-populated fields.
    """
    # Input validation
    if not isinstance(analysis, dict):
        logger.error("_merge_manifest_into_analysis: analysis is not a dict, returning as-is")
        return analysis if isinstance(analysis, dict) else {}

    if not isinstance(convo_dir, Path):
        logger.error("_merge_manifest_into_analysis: convo_dir is not a Path")
        return analysis

    if raw_thread_text is None:
        logger.warning("_merge_manifest_into_analysis: raw_thread_text is None, using empty string")
        raw_thread_text = ""

    manifest = _read_manifest(convo_dir)

    # Always normalize subject even when already present
    existing_subject = analysis.get("subject", "")
    if not isinstance(existing_subject, str):
        existing_subject = ""

    # Build subject candidates, preferring existing
    subj_candidates: list[str] = []
    if existing_subject and existing_subject != "Email thread":
        subj_candidates.append(existing_subject)

    # Add manifest candidates
    if isinstance(manifest, dict):
        smart = (manifest.get("smart_subject") or "").strip()
        if smart:
            subj_candidates.append(smart)
        subj = (manifest.get("subject") or "").strip()
        if subj:
            subj_candidates.append(subj)

    # Parse from raw headers
    md = extract_email_metadata(raw_thread_text or "")
    if isinstance(md, dict) and md.get("subject"):
        subj_candidates.append(str(md["subject"]).strip())

    # Always normalize and cap
    final_subject = subj_candidates[0] if subj_candidates else "Email thread"
    analysis["subject"] = _safe_str(_normalize_subject_line(final_subject), SUBJECT_MAX_LEN)

    # Participants: UNION merge (not just if empty)
    existing_participants = analysis.get("participants", [])
    if not isinstance(existing_participants, list):
        existing_participants = []

    manifest_participants = _participants_from_manifest(manifest) if manifest else []

    # Union with de-duplication
    seen_keys: set[str] = set()
    merged_participants: list[dict[str, str]] = []

    # Add existing participants first
    for p in existing_participants:
        if isinstance(p, dict):
            email_key = (p.get("email") or "").lower()
            name_key = _normalize_name(p.get("name", ""))
            if not email_key and not name_key:
                continue
            key = email_key or f"name:{name_key}"
            if key not in seen_keys:
                seen_keys.add(key)
                merged_participants.append(p)

    # Add new participants from manifest
    for p in manifest_participants:
        if isinstance(p, dict):
            email_key = (p.get("email") or "").lower()
            name_key = _normalize_name(p.get("name", ""))
            if not email_key and not name_key:
                continue
            key = email_key or f"name:{name_key}"
            if key not in seen_keys:
                seen_keys.add(key)
                merged_participants.append(p)

    analysis["participants"] = merged_participants[:MAX_PARTICIPANTS]

    # Key dates: ALWAYS add start/end from manifest, union with existing
    fl = analysis.get("facts_ledger", {}) or {}
    if not isinstance(fl, dict):
        fl = {}

    existing_dates = fl.get("key_dates", [])
    if not isinstance(existing_dates, list):
        existing_dates = []

    # Collect manifest dates
    manifest_dates: list[dict[str, str]] = []
    if isinstance(manifest, dict):
        try:
            time_span = manifest.get("time_span") or {}
            start = time_span.get("start_local") or time_span.get("start")
            end = time_span.get("end_local") or time_span.get("end")
            if start:
                manifest_dates.append(
                    {
                        "date": str(start),
                        "event": "Conversation start",
                        "importance": "reference",
                    }
                )
            if end:
                manifest_dates.append(
                    {
                        "date": str(end),
                        "event": "Conversation end",
                        "importance": "reference",
                    }
                )
        except Exception:
            pass

    # Union dates with de-duplication by (date, event) pair
    seen_date_keys: set[tuple[str, str]] = set()
    merged_dates: list[dict[str, str]] = []

    # Add existing dates first
    for d in existing_dates:
        if isinstance(d, dict):
            date_key = (str(d.get("date", "")).lower(), str(d.get("event", "")).lower())
            if date_key not in seen_date_keys:
                seen_date_keys.add(date_key)
                merged_dates.append(d)

    # Add manifest dates
    for d in manifest_dates:
        date_key = (str(d.get("date", "")).lower(), str(d.get("event", "")).lower())
        if date_key not in seen_date_keys:
            seen_date_keys.add(date_key)
            merged_dates.append(d)

    fl["key_dates"] = merged_dates[:MAX_FACT_ITEMS]
    analysis["facts_ledger"] = fl

    return analysis


def _union_analyses(improved: dict[str, Any], initial: dict[str, Any], catalog: list[str]) -> dict[str, Any]:
    """
    Union improved and initial analyses to avoid dropping valid content.
    Improved items come first, then unique items from initial.

    HIGH #50: Fixed data loss bug - ensure initial data is preserved when improved is empty.
    """
    # HIGH #50: Start with a safe base - use initial if improved is empty/invalid
    if not improved or not isinstance(improved, dict):
        result = dict(initial) if isinstance(initial, dict) else {}
    else:
        result = dict(improved)

    # Helper to normalize keys for de-duplication
    def _norm_key(s: str) -> str:
        return s.strip().lower() if isinstance(s, str) else ""

    # HIGH #50: Union participants - handle case where improved has no participants
    initial_participants = initial.get("participants", []) if isinstance(initial, dict) else []
    improved_participants = result.get("participants", []) if isinstance(result, dict) else []

    if isinstance(initial_participants, list):
        # Build participant mapping starting with improved items
        all_participants = {}

        # Add improved participants first (if any)
        for p in improved_participants:
            if isinstance(p, dict):
                email_key = _norm_key(p.get("email") or "")
                name_key = _normalize_name(p.get("name") or "")
                if email_key or name_key:
                    key = email_key or f"name:{name_key}"
                    all_participants[key] = p

        # Add initial participants that aren't already present
        for p in initial_participants:
            if isinstance(p, dict):
                email_key = _norm_key(p.get("email") or "")
                name_key = _normalize_name(p.get("name") or "")
                if email_key or name_key:
                    key = email_key or f"name:{name_key}"
                    if key not in all_participants:  # Only add if not already present
                        all_participants[key] = p

        result["participants"] = list(all_participants.values())[:MAX_PARTICIPANTS]

    # Union summary points
    initial_summary = initial.get("summary", [])
    if isinstance(initial_summary, list):
        improved_summary = result.get("summary", [])
        seen_summary = {_norm_key(s) for s in improved_summary if isinstance(s, str)}
        merged_summary = list(improved_summary)
        merged_summary.extend([s for s in initial_summary if isinstance(s, str) and _norm_key(s) not in seen_summary])
        result["summary"] = merged_summary[:MAX_SUMMARY_POINTS]

    # Union risk_indicators
    initial_risks = initial.get("risk_indicators", [])
    if isinstance(initial_risks, list):
        improved_risks = result.get("risk_indicators", [])
        seen_risks = {_norm_key(r) for r in improved_risks if isinstance(r, str)}
        merged_risks = list(improved_risks)
        merged_risks.extend([r for r in initial_risks if isinstance(r, str) and _norm_key(r) not in seen_risks])
        result["risk_indicators"] = merged_risks

    # Union facts_ledger items
    improved_fl = result.get("facts_ledger", {})
    initial_fl = initial.get("facts_ledger", {})

    if isinstance(improved_fl, dict) and isinstance(initial_fl, dict):
        for field_name in [
            "known_facts",
            "required_for_resolution",
            "what_we_have",
            "what_we_need",
            "materiality_for_company",
            "materiality_for_me",
        ]:
            initial_list = initial_fl.get(field_name, [])
            if isinstance(initial_list, list):
                improved_list = improved_fl.get(field_name, [])
                seen_items = {_norm_key(i) for i in improved_list if isinstance(i, str)}
                merged_list = list(improved_list)
                merged_list.extend(
                    [item for item in initial_list if isinstance(item, str) and _norm_key(item) not in seen_items]
                )
                improved_fl[field_name] = merged_list[:MAX_FACT_ITEMS]

        # Union commitments_made
        initial_commits = initial_fl.get("commitments_made", [])
        if isinstance(initial_commits, list):
            improved_commits = improved_fl.get("commitments_made", [])
            # De-dupe based on a tuple of normalized values
            all_commits = {
                (_norm_key(c.get("by") or ""), _norm_key(c.get("commitment") or "")): c
                for c in improved_commits
                if isinstance(c, dict)
            }
            initial_commits_dict = {
                (_norm_key(c.get("by") or ""), _norm_key(c.get("commitment") or "")): c
                for c in initial_commits
                if isinstance(c, dict)
            }
            all_commits.update({k: v for k, v in initial_commits_dict.items() if k not in all_commits})
            improved_fl["commitments_made"] = list(all_commits.values())[:MAX_FACT_ITEMS]

        # Union key_dates
        initial_dates = initial_fl.get("key_dates", [])
        if isinstance(initial_dates, list):
            improved_dates = improved_fl.get("key_dates", [])
            all_dates = {
                (_norm_key(d.get("date") or ""), _norm_key(d.get("event") or "")): d for d in improved_dates if isinstance(d, dict)
            }
            initial_dates_dict = {
                (_norm_key(d.get("date") or ""), _norm_key(d.get("event") or "")): d for d in initial_dates if isinstance(d, dict)
            }
            all_dates.update({k: v for k, v in initial_dates_dict.items() if k not in all_dates})
            improved_fl["key_dates"] = list(all_dates.values())[:MAX_FACT_ITEMS]

        result["facts_ledger"] = improved_fl

    # Union next_actions
    initial_actions = initial.get("next_actions", [])
    if isinstance(initial_actions, list):
        improved_actions = result.get("next_actions", [])
        all_actions = {
            (_norm_key(a.get("owner") or ""), _norm_key(a.get("action") or "")): a for a in improved_actions if isinstance(a, dict)
        }
        initial_actions_dict = {
            (_norm_key(a.get("owner") or ""), _norm_key(a.get("action") or "")): a for a in initial_actions if isinstance(a, dict)
        }
        all_actions.update({k: v for k, v in initial_actions_dict.items() if k not in all_actions})
        result["next_actions"] = list(all_actions.values())[:MAX_NEXT_ACTIONS]

    # Re-apply normalization to ensure all caps and coercions are applied
    return _normalize_analysis(result, catalog)


async def _retry(callable_fn, *args, retries: int = 2, delay: float = 0.5, **kwargs):
    """
    Retry helper that supports both synchronous and asynchronous callables.
    Uses exponential backoff with jitter and asyncio-friendly sleep.
    """

    attempt = 0
    max_retries = retries if retries is not None else 2
    base_delay = delay if delay is not None else 0.5
    while True:
        try:
            result = callable_fn(*args, **kwargs)
            if inspect.isawaitable(result):
                return await result
            return result
        except Exception as e:
            attempt += 1
            if attempt > max_retries:
                raise
            # Exponential backoff with ±20% jitter
            backoff = base_delay * (2 ** (attempt - 1))
            jitter = backoff * 0.2
            sleep_for = backoff + (random.random() * 2 * jitter - jitter)
            logger.warning(
                "LLM call failed (%s). Retrying %d/%d after %.2fs ...",
                e,
                attempt,
                max_retries,
                sleep_for,
            )
            await asyncio.sleep(sleep_for)


def _calc_max_output_tokens() -> int:
    """
    Calculate dynamic token budget based on configured caps.
    """
    base = 600
    budget = (
        base
        + 4 * MAX_SUMMARY_POINTS
        + 20 * MAX_PARTICIPANTS
        + 24 * MAX_NEXT_ACTIONS
        + 20 * MAX_FACT_ITEMS * 5  # 5 ledgers
    )
    # Clamp to reasonable range
    return max(1200, min(3500, budget))


def _llm_routing_kwargs(provider: str) -> dict[str, Any]:
    """
    Build optional LLM routing kwargs based on provider and environment.
    Returns empty dict if routing is not possible.
    """
    kwargs: dict[str, Any] = {}

    # Try to add provider
    kwargs["provider"] = provider

    # Check for environment-based model configuration
    provider_upper = provider.upper()
    model_env_var = f"SUMMARIZER_MODEL_{provider_upper}"
    model = os.getenv(model_env_var)
    if model:
        kwargs["model"] = model

    return kwargs


async def analyze_email_thread_with_ledger(
    thread_text: str,
    catalog: list[str] = DEFAULT_CATALOG,
    provider: str = "vertex",
    temperature: float = 0.2,
) -> dict[str, Any]:
    """
    Analyze an email thread using a "facts ledger" approach.

    Returns:
        A dict with required fields:
          category, subject, participants, facts_ledger, summary, next_actions, risk_indicators
        plus internal _metadata.
    """
    # Ensure non-empty catalog for schema enum
    catalog = catalog or DEFAULT_CATALOG

    # Sanitize defensively (callers outside CLI may pass raw text)
    cleaned_thread = clean_email_text(thread_text or "")

    # Calculate dynamic token budget
    max_tokens = _calc_max_output_tokens()

    # Get routing kwargs
    routing_kwargs = _llm_routing_kwargs(provider)

    # Import inspect for signature checking

    # Check which routing kwargs are accepted by complete_json
    try:
        cj_sig = inspect.signature(complete_json)
        cj_params = set(cj_sig.parameters.keys())
        cj_routing = {k: v for k, v in routing_kwargs.items() if k in cj_params}
        routing_applied = bool(cj_routing)
    except Exception:
        cj_routing = {}
        routing_applied = False

    # Add comprehensive debug logging for LLM calls
    if routing_applied:
        logger.info(
            "Starting email thread analysis: %d chars, %d categories, provider=%s used for routing, temp=%.2f, token_budget=%d",
            len(cleaned_thread),
            len(catalog),
            provider,
            temperature,
            max_tokens,
        )
    else:
        logger.info(
            "Starting email thread analysis: %d chars, %d categories, provider=%s stored in metadata only, temp=%.2f, token_budget=%d",
            len(cleaned_thread),
            len(catalog),
            provider,
            temperature,
            max_tokens,
        )

    if not cleaned_thread.strip():
        now = datetime.now(UTC).isoformat()
        return {
            "category": catalog[-1],
            "subject": "Email thread",
            "participants": [],
            "facts_ledger": {
                "known_facts": [],
                "key_dates": [],
                "commitments_made": [],
                "required_for_resolution": [],
                "what_we_have": [],
                "what_we_need": ["No thread content provided"],
                "materiality_for_company": [],
                "materiality_for_me": [],
            },
            "summary": [],
            "next_actions": [],
            "risk_indicators": [],
            "_metadata": {
                "analyzed_at": now,
                "provider": provider,
                "completeness_score": 0,
                "version": ANALYZER_VERSION,
                "input_chars": 0,
            },
        }

    # Schema used for structured output
    response_schema = {
        "type": "object",
        "properties": {
            "category": {
                "type": "string",
                "enum": catalog,
                "description": "Primary category this thread belongs to",
            },
            "subject": {
                "type": "string",
                "description": "Short descriptive title (max 100 chars)",
            },
            "participants": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "role": {
                            "type": "string",
                            "enum": [
                                "client",
                                "broker",
                                "underwriter",
                                "internal",
                                "other",
                            ],
                        },
                        "email": {"type": "string"},
                        "tone": {
                            "type": "string",
                            "enum": [
                                "professional",
                                "frustrated",
                                "urgent",
                                "friendly",
                                "demanding",
                                "neutral",
                            ],
                        },
                        "stance": {
                            "type": "string",
                            "description": "Their position/attitude in this thread",
                        },
                    },
                    "required": ["name", "role", "tone", "stance"],
                },
            },
            "facts_ledger": {
                "type": "object",
                "properties": {
                    "known_facts": {"type": "array", "items": {"type": "string"}},
                    "key_dates": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "date": {"type": "string"},
                                "event": {"type": "string"},
                                "importance": {
                                    "type": "string",
                                    "enum": ["critical", "important", "reference"],
                                },
                            },
                            "required": ["date", "event", "importance"],
                        },
                    },
                    "commitments_made": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "by": {"type": "string"},
                                "commitment": {"type": "string"},
                                "deadline": {"type": "string"},
                                "feasibility": {
                                    "type": "string",
                                    "enum": [
                                        "achievable",
                                        "challenging",
                                        "risky",
                                        "impossible",
                                    ],
                                },
                            },
                            "required": ["by", "commitment", "feasibility"],
                        },
                    },
                    "required_for_resolution": {"type": "array", "items": {"type": "string"}},
                    "what_we_have": {"type": "array", "items": {"type": "string"}},
                    "what_we_need": {"type": "array", "items": {"type": "string"}},
                    "materiality_for_company": {"type": "array", "items": {"type": "string"}},
                    "materiality_for_me": {"type": "array", "items": {"type": "string"}},
                },
                "required": [
                    "known_facts",
                    "key_dates",
                    "commitments_made",
                    "required_for_resolution",
                    "what_we_have",
                    "what_we_need",
                    "materiality_for_company",
                    "materiality_for_me",
                ],
            },
            "summary": {"type": "array", "items": {"type": "string"}},
            "next_actions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "owner": {"type": "string"},
                        "action": {"type": "string"},
                        "due": {"type": "string"},
                        "status": {
                            "type": "string",
                            "enum": ["open", "in_progress", "blocked", "completed"],
                        },
                        "priority": {
                            "type": "string",
                            "enum": ["critical", "high", "medium", "low"],
                        },
                    },
                    "required": ["owner", "action", "status", "priority"],
                },
            },
            "risk_indicators": {"type": "array", "items": {"type": "string"}},
        },
        "required": [
            "category",
            "subject",
            "participants",
            "facts_ledger",
            "summary",
            "next_actions",
            "risk_indicators",
        ],
    }

    # Stop sequences:
    # For JSON outputs, we avoid stop sequences entirely to prevent truncation inside JSON values.
    json_stop_sequences: list[str] | None = None
    # For plain text fallback, keep empty to avoid mid-sentence truncation as well.
    text_stop_sequences: list[str] | None = None

    # System prompt
    system = """You are a senior insurance account manager analyzing email threads with extreme attention to detail.

Your analysis must capture the COMPLETE FACTS LEDGER including:
1. KNOWN FACTS: Key confirmed facts and statements.
2. KEY DATES: Critical deadlines and events.
3. COMMITMENTS MADE: What was promised, by whom, and feasibility.
4. REQUIRED FOR RESOLUTION: The essential next steps or information needed to resolve the thread.
5. WHAT WE HAVE: Information or documents we possess.
6. WHAT WE NEED: Information or documents we must obtain.
7. MATERIALITY FOR COMPANY: Why this thread is important for the company.
8. MATERIALITY FOR ME: Why this thread is important for me, the user.

CRITICAL RULES:
- Extract ONLY facts from the email text - no assumptions
- Identify every participant's emotional state and urgency
- Flag any promises that may be problematic
- Note all missing information that would be needed for decisions
- Be specific about who said what
- Identify risks and red flags"""

    # Compose user prompt with bounded sizes
    user = f"""Analyze this email thread and extract a comprehensive facts ledger:

Thread:
{cleaned_thread[:MAX_THREAD_CHARS]}

Create a detailed analysis following the schema. Be thorough in identifying:
- All known facts and key dates.
- All commitments made by any party.
- What is required for resolution.
- What information we have versus what we need.
- The materiality of this thread for the company and for me.

Output valid JSON matching the required schema."""

    # --- Pass 1: Initial analysis (robust JSON parsing) ---
    initial_analysis = {}  # Initialize to avoid unbound variable errors
    try:
        _cj_kwargs = {
            "max_output_tokens": max_tokens,
            "temperature": temperature,
            "response_schema": response_schema,
        }
        if json_stop_sequences:
            _cj_kwargs["stop_sequences"] = json_stop_sequences
        # Add routing kwargs if supported
        _cj_kwargs.update(cj_routing)

        initial_response = await _retry(
            complete_json,
            system,
            user,
            **_cj_kwargs,
        )
        try:
            parsed = _try_load_json(initial_response)
        except ValueError as parse_error:
            logger.warning("Initial JSON parse failed: %s. Attempting recovery.", parse_error)
            raise

        initial_analysis = _normalize_analysis(parsed, catalog)

        # Log initial analysis results
        logger.info(
            "Initial analysis complete: %d participants, %d summary points, %d next actions",
            len(initial_analysis.get("participants", [])),
            len(initial_analysis.get("summary", [])),
            len(initial_analysis.get("next_actions", [])),
        )
    except Exception as e:
        logger.error("Structured analysis failed: %s. Falling back to text mode with retry.", e)

        # Retry with text mode (improved error recovery)
        retry_attempts = 3
        last_error = None

        # Check which routing kwargs are accepted by complete_text
        try:
            ct_sig = inspect.signature(complete_text)
            ct_params = set(ct_sig.parameters.keys())
            ct_routing = {k: v for k, v in routing_kwargs.items() if k in ct_params}
        except Exception:
            ct_routing = {}

        for attempt in range(retry_attempts):
            try:
                logger.debug("Text mode recovery attempt %d/%d", attempt + 1, retry_attempts)
                _ct_kwargs = {
                    "max_output_tokens": max_tokens,
                    "temperature": temperature + (0.1 * attempt),  # Increase temp slightly on retries
                }
                if text_stop_sequences:
                    _ct_kwargs["stop_sequences"] = text_stop_sequences
                # Add routing kwargs if supported
                _ct_kwargs.update(ct_routing)

                fb = await _retry(
                    complete_text,
                    system,
                    user + "\n\nIMPORTANT: Output must be valid JSON matching the schema.",
                    **_ct_kwargs,
                )

                # Try to parse the fallback response
                try:
                    parsed_fb = _try_load_json(fb)
                except ValueError:
                    # If JSON parsing fails, use the raw text
                    logger.debug("Text mode JSON parse failed on attempt %d", attempt + 1)
                    parsed_fb = {}

                initial_analysis = _normalize_analysis(parsed_fb, catalog)

                # Validate we got something useful
                if initial_analysis.get("summary") or initial_analysis.get("participants"):
                    logger.info("Text mode recovery successful on attempt %d", attempt + 1)
                    break

            except Exception as retry_error:
                last_error = retry_error
                logger.warning("Text mode attempt %d failed: %s", attempt + 1, retry_error)
                if attempt < retry_attempts - 1:

                    await asyncio.sleep(1.0 * (attempt + 1))  # Exponential backoff

        # Ensure we have initial_analysis even if all attempts failed
        if not initial_analysis:
            initial_analysis = _normalize_analysis({}, catalog)

        # If we still don't have a good analysis, create a minimal one
        if not initial_analysis.get("summary"):
            logger.error("All recovery attempts failed. Creating minimal analysis.")
            initial_analysis["summary"] = [
                "Failed to fully parse email thread. Raw content preserved for manual review."
            ]
            initial_analysis["facts_ledger"] = initial_analysis.get("facts_ledger", {})
            fl = initial_analysis["facts_ledger"]
            if isinstance(fl, dict):
                what_we_need = fl.get("what_we_need", [])
                if isinstance(what_we_need, list):
                    what_we_need.append(f"Automated parsing failed: {last_error or 'Unknown error'}")
                    fl["what_we_need"] = what_we_need

    # --- Pass 2: Critic review for completeness/accuracy ---
    critic_system = """You are a quality control specialist reviewing email thread analyses.

Your job is to verify:
1. All participants are correctly identified with accurate tone/stance
2. No explicit asks or commitments were missed
3. All unknowns and information gaps are captured
4. Forbidden promises are properly identified
5. Risk indicators are comprehensive
6. The analysis is factual, not assumptive"""

    critic_schema = {
        "type": "object",
        "properties": {
            "missed_items": {
                "type": "object",
                "properties": {
                    "participants": {"type": "array", "items": {"type": "string"}},
                    "requests": {"type": "array", "items": {"type": "string"}},
                    "commitments": {"type": "array", "items": {"type": "string"}},
                    "unknowns": {"type": "array", "items": {"type": "string"}},
                    "risks": {"type": "array", "items": {"type": "string"}},
                },
            },
            "accuracy_issues": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "field": {"type": "string"},
                        "issue": {"type": "string"},
                        "correction": {"type": "string"},
                    },
                },
            },
            "completeness_score": {"type": "integer", "minimum": 0, "maximum": 100},
            "critical_gaps": {"type": "array", "items": {"type": "string"}},
        },
        "required": [
            "missed_items",
            "accuracy_issues",
            "completeness_score",
            "critical_gaps",
        ],
    }

    critic_user = f"""Review this email thread analysis for completeness and accuracy:

Original Thread (first {CRITIC_THREAD_CHARS} chars):
{cleaned_thread[:CRITIC_THREAD_CHARS]}

Analysis to Review:
{json.dumps(initial_analysis, ensure_ascii=False, indent=2)}

Check for:
- Any participants not captured or mischaracterized
- Missed requests, commitments, or deadlines
- Information gaps not identified
- Risks or red flags overlooked
- Inaccurate tone/stance assessments"""

    try:
        _crit_kwargs = {
            "max_output_tokens": 1000,  # Keep critic pass smaller
            "temperature": 0.1,
            "response_schema": critic_schema,
        }
        if json_stop_sequences:
            _crit_kwargs["stop_sequences"] = json_stop_sequences
        # Add routing kwargs if supported
        _crit_kwargs.update(cj_routing)

        critic_response = await _retry(
            complete_json,
            critic_system,
            critic_user,
            **_crit_kwargs,
        )
        critic_feedback = _try_load_json(critic_response) or {
            "missed_items": {},
            "accuracy_issues": [],
            "completeness_score": 80,
            "critical_gaps": [],
        }
    except Exception as e:
        logger.warning("Failed to get critic feedback: %s", e)
        critic_feedback = {
            "missed_items": {},
            "accuracy_issues": [],
            "completeness_score": 80,
            "critical_gaps": [],
        }

    # --- Pass 3: Improvement if the critic flags gaps ---
    final_analysis = initial_analysis
    # Safely coerce completeness score to int for comparisons/metadata
    _raw_score = critic_feedback.get("completeness_score", 100)
    try:
        _score_int = int(_raw_score)
    except (TypeError, ValueError):
        _score_int = 100
    _critical_gaps = critic_feedback.get("critical_gaps") or []

    if _score_int < 85 or bool(_critical_gaps):
        improvement_system = """You are an expert analyst improving an email thread analysis based on feedback.

Add any missed information while maintaining accuracy. Do not remove correct information."""

        improvement_user = f"""Improve this analysis based on the feedback:

Current Analysis:
{json.dumps(initial_analysis, ensure_ascii=False, indent=2)}

Feedback:
{json.dumps(critic_feedback, ensure_ascii=False, indent=2)}

Original Thread (for reference, first {IMPROVE_THREAD_CHARS} chars):
{cleaned_thread[:IMPROVE_THREAD_CHARS]}

Generate an improved analysis that addresses all feedback while maintaining the same schema."""

        try:
            # Improvement pass gets 10% more token budget
            improve_tokens = int(max_tokens * 1.1)
            _imp_kwargs = {
                "max_output_tokens": improve_tokens,
                "temperature": 0.2,
                "response_schema": response_schema,
            }
            if json_stop_sequences:
                _imp_kwargs["stop_sequences"] = json_stop_sequences
            # Add routing kwargs if supported
            _imp_kwargs.update(cj_routing)

            improved_response = await _retry(
                complete_json,
                improvement_system,
                improvement_user,
                **_imp_kwargs,
            )
            parsed_imp = _try_load_json(improved_response)
            if parsed_imp:
                # Union improved with initial to avoid dropping valid content
                improved_analysis = _normalize_analysis(parsed_imp, catalog)
                final_analysis = _union_analyses(improved_analysis, initial_analysis, catalog)
                logger.debug("Improved analysis applied with union merge (completeness score: %d)", _score_int)
        except Exception as e:
            logger.warning("Failed to get improved analysis: %s", e)

    # Attach metadata (stable; helpful for auditability)
    final_analysis["_metadata"] = {
        "analyzed_at": datetime.now(UTC).isoformat(),
        "provider": provider,
        "completeness_score": _score_int,
        "version": ANALYZER_VERSION,
        "input_chars": len(cleaned_thread),
    }
    return final_analysis


# =============================================================================
# Higher-level API: analyze a conversation directory (read, analyze, enrich)
# =============================================================================
async def analyze_conversation_dir(
    thread_dir: Path,
    catalog: list[str] = DEFAULT_CATALOG,
    provider: str = os.getenv("EMBED_PROVIDER", "vertex"),
    temperature: float = 0.2,
    merge_manifest: bool = True,
) -> dict[str, Any]:
    """
    Read Conversation.txt from `thread_dir`, run the facts-ledger analysis, and
    (optionally) merge manifest metadata (subject, participants, dates).
    Returns the final analysis dict.
    """
    convo = Path(thread_dir).expanduser().resolve()
    convo_txt = convo / "Conversation.txt"
    if not convo_txt.exists():
        raise FileNotFoundError(f"Conversation.txt not found in {convo}")

    raw = read_text_file(convo_txt)
    cleaned = clean_email_text(raw)

    data = await analyze_email_thread_with_ledger(
        thread_text=cleaned,
        catalog=(catalog or DEFAULT_CATALOG),
        provider=provider,
        temperature=temperature,
    )

    if merge_manifest:
        data = _merge_manifest_into_analysis(data, convo, raw)

    # Safety: enforce caps again in case merge added items
    data = _normalize_analysis(data, (catalog or DEFAULT_CATALOG))

    return data


def format_analysis_as_markdown(analysis: dict[str, Any]) -> str:
    """
    Format an email thread analysis as Markdown text.
    Optimized with StringIO for better performance.

    Args:
        analysis: Analysis dictionary from analyze_email_thread_with_ledger()

    Returns:
        Formatted markdown string
    """

    # Use StringIO for efficient string building
    buffer = StringIO()

    # Header
    buffer.write("# Email Thread Analysis\n\n")

    # Summary section
    buffer.write("## Summary\n\n")
    buffer.write(f"**Category**: {analysis.get('category', 'unknown')}  \n")
    buffer.write(f"**Subject**: {analysis.get('subject', 'No subject')}\n\n")

    # Overview
    buffer.write("### Overview\n\n")
    for point in analysis.get("summary", []):
        buffer.write(f"- {_md_escape(point)}\n")

    # Participants
    buffer.write("\n## Participants\n\n")
    for p in analysis.get("participants", []):
        if isinstance(p, dict):
            buffer.write(
                f"- **{_md_escape(p.get('name', ''))}** ({_md_escape(p.get('role', ''))})\n"
                f"  - Email: {_md_escape(p.get('email', ''))}\n"
                f"  - Tone: {_md_escape(p.get('tone', ''))}\n"
                f"  - Stance: {_md_escape(p.get('stance', ''))}\n"
            )

    # Facts Ledger
    buffer.write("\n## Facts Ledger\n\n")

    # Known Facts
    buffer.write("### Known Facts\n\n")
    for fact in analysis.get("facts_ledger", {}).get("known_facts", []):
        buffer.write(f"- {_md_escape(fact)}\n")

    # Key Dates
    buffer.write("\n### Key Dates\n\n")
    for kd in analysis.get("facts_ledger", {}).get("key_dates", []):
        if isinstance(kd, dict):
            buffer.write(
                f"- **{_md_escape(kd.get('date', ''))}**: {_md_escape(kd.get('event', ''))} "
                f"({_md_escape(kd.get('importance', ''))})\n"
            )

    # Commitments Made
    buffer.write("\n### Commitments Made\n\n")
    for commit in analysis.get("facts_ledger", {}).get("commitments_made", []):
        if isinstance(commit, dict):
            buffer.write(
                f"- **By**: {_md_escape(commit.get('by', ''))}\n"
                f"  - **Commitment**: {_md_escape(commit.get('commitment', ''))}\n"
                f"  - **Deadline**: {_md_escape(commit.get('deadline', ''))}\n"
                f"  - **Feasibility**: {_md_escape(commit.get('feasibility', ''))}\n\n"
            )

    # Resolution
    buffer.write("### Resolution\n\n")
    buffer.write("#### Required for Resolution\n")
    for item in analysis.get("facts_ledger", {}).get("required_for_resolution", []):
        buffer.write(f"- {_md_escape(item)}\n")

    buffer.write("\n#### What We Have\n")
    for item in analysis.get("facts_ledger", {}).get("what_we_have", []):
        buffer.write(f"- {_md_escape(item)}\n")

    buffer.write("\n#### What We Need\n")
    for item in analysis.get("facts_ledger", {}).get("what_we_need", []):
        buffer.write(f"- {_md_escape(item)}\n")

    # Materiality
    buffer.write("\n### Materiality\n\n")
    buffer.write("#### For Company\n")
    for item in analysis.get("facts_ledger", {}).get("materiality_for_company", []):
        buffer.write(f"- {_md_escape(item)}\n")

    buffer.write("\n#### For Me\n")
    for item in analysis.get("facts_ledger", {}).get("materiality_for_me", []):
        buffer.write(f"- {_md_escape(item)}\n")

    # Next Actions
    buffer.write("\n## Next Actions\n\n")
    for action in analysis.get("next_actions", []):
        if isinstance(action, dict):
            buffer.write(
                f"- **{_md_escape(action.get('owner', ''))}**: {_md_escape(action.get('action', ''))}\n"
                f"  - Priority: {_md_escape(action.get('priority', ''))}\n"
                f"  - Status: {_md_escape(action.get('status', ''))}\n"
            )
            if action.get("due"):
                buffer.write(f"  - Due: {_md_escape(action.get('due', ''))}\n")
            buffer.write("\n")

    # Risk Indicators
    buffer.write("## Risk Indicators\n\n")
    for risk in analysis.get("risk_indicators", []):
        buffer.write(f"- 🚨 {_md_escape(risk)}\n")

    # Metadata
    if "_metadata" in analysis:
        meta = analysis["_metadata"]
        buffer.write("\n---\n\n")
        buffer.write("## Metadata\n\n")
        buffer.write(f"- **Analyzed at**: {meta.get('analyzed_at', 'N/A')}\n")
        buffer.write(f"- **Provider**: {meta.get('provider', 'N/A')}\n")
        buffer.write(f"- **Completeness score**: {meta.get('completeness_score', 0)}%\n")
        buffer.write(f"- **Version**: {meta.get('version', 'N/A')}\n")

    return buffer.getvalue()


# =============================================================================
# Filesystem utilities
# =============================================================================
def _atomic_write_text(path: Path, content: str, encoding: str = "utf-8") -> None:
    """Write text atomically with basic cross-platform resilience."""
    ensure_dir(path.parent)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding=encoding) as tmp:
            tmp.write(content)
        # Retry with exponential backoff to handle transient FS/AV locks
        delay = 0.05
        for attempt in range(5):
            try:
                Path(tmp_name).replace(path)
                return
            except OSError:
                if attempt == 4:
                    raise
                time.sleep(delay)
                delay *= 2
    finally:
        try:
            if Path(tmp_name).exists():
                Path(tmp_name).unlink()
        except Exception:
            pass


def _safe_csv_cell(x: Any) -> str:
    """Guard against CSV/Excel formula injection (handles leading whitespace/tab)."""
    s = "" if x is None else str(x)
    stripped = s.lstrip()
    if (stripped and stripped[0] in ("=", "+", "-", "@")) or s.startswith("\t"):
        return "'" + s
    return s


def _append_todos_csv(root: Path, thread_name: str, todos: list[dict[str, Any]]) -> None:
    """
    Append next_actions to root todo.csv with basic deduplication (owner+what+thread).
    Uses DictReader/Writer to avoid comma-splitting pitfalls.
    """
    out = root / "todo.csv"
    ensure_dir(out.parent)

    existing_keys = set()
    if out.exists():
        try:
            with out.open("r", newline="", encoding="utf-8") as f:
                r = csv.DictReader(f)
                for row in r:
                    who = (row.get("who") or "").strip().lower()
                    what = (row.get("what") or "").strip().lower()
                    thread = (row.get("thread") or "").strip().lower()
                    if who or what or thread:
                        existing_keys.add((who, what, thread))
        except Exception:
            # best-effort; continue without existing keys
            pass

    # Open in append mode and write header if file didn't exist or was empty
    write_header = not out.exists() or out.stat().st_size == 0
    with out.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["who", "what", "due", "status", "priority", "thread"])
        if write_header:
            w.writeheader()
        for t in todos or []:
            if not isinstance(t, dict):
                continue
            who = t.get("owner", "")
            what = t.get("action", "")
            key = (
                str(who).strip().lower(),
                str(what).strip().lower(),
                thread_name.strip().lower(),
            )
            if key in existing_keys:
                continue
            w.writerow(
                {
                    "who": _safe_csv_cell(who),
                    "what": _safe_csv_cell(what),
                    "due": _safe_csv_cell(t.get("due", "")),
                    "status": _safe_csv_cell(t.get("status", "open")),
                    "priority": _safe_csv_cell(t.get("priority", "medium")),
                    "thread": _safe_csv_cell(thread_name),
                }
            )
            existing_keys.add(key)  # avoid duplicates within the same run


# =============================================================================
# CLI
# =============================================================================
async def async_main():
    # Configure logging for CLI entry point - ensure module logger propagates
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # Ensure the module logger propagates to root logger
    logger.propagate = True
    logger.setLevel(logging.INFO)

    ap = argparse.ArgumentParser(
        description="Summarize a single email thread directory with comprehensive facts ledger."
    )
    ap.add_argument(
        "--thread",
        required=True,
        help="Path to a conversation directory containing Conversation.txt",
    )
    ap.add_argument("--catalog", nargs="*", default=DEFAULT_CATALOG, help="Allowed categories")
    ap.add_argument(
        "--write_todos_csv",
        action="store_true",
        help="Append next_actions to root todo.csv",
    )
    ap.add_argument(
        "--provider",
        default=os.getenv("EMBED_PROVIDER", "vertex"),
        help="Provider to use for LLM calls. If emailops.llm_client supports provider/model kwargs, this value is routed "
        "otherwise it is stored in metadata. Defaults to $EMBED_PROVIDER or 'vertex'.",
    )
    ap.add_argument("--temperature", type=float, default=0.2, help="Temperature for generation")
    ap.add_argument(
        "--output-format",
        choices=["json", "markdown"],
        default="json",
        help='Write JSON by default. With "markdown", also write summary.md. Combine with --no-json to skip JSON.',
    )
    ap.add_argument(
        "--no-json",
        action="store_true",
        help="If set with --output-format markdown, skip writing JSON",
    )
    ap.add_argument(
        "--no-manifest-merge",
        action="store_true",
        help="Do NOT merge manifest metadata (subject/participants/dates)",
    )
    args = ap.parse_args()

    # Guard: if user passed --catalog with no values, fall back to defaults
    if not args.catalog:
        args.catalog = DEFAULT_CATALOG

    tdir = Path(args.thread).expanduser().resolve()
    convo = tdir / "Conversation.txt"
    if not convo.exists():
        raise SystemExit(f"Conversation.txt not found in {tdir}")

    # Enhanced analysis + (optional) manifest enrichment
    try:
        data = await analyze_conversation_dir(
            thread_dir=tdir,
            catalog=args.catalog,
            provider=args.provider,
            temperature=args.temperature,
            merge_manifest=(not args.no_manifest_merge),
        )
    except Exception as e:
        raise SystemExit(f"Failed to analyze thread: {e}") from e

    # Save JSON (atomically)
    write_json = not (args.output_format == "markdown" and args.no_json)
    if write_json:
        out_json = tdir / "summary.json"
        _atomic_write_text(out_json, json.dumps(data, ensure_ascii=False, indent=2))
        logger.info("Wrote %s", out_json)
    else:
        logger.info("Skipped JSON output (--no-json with markdown)")

    # Optionally save as markdown for readability (using the existing function)
    if args.output_format == "markdown":
        markdown_content = format_analysis_as_markdown(data)
        out_md = tdir / "summary.md"
        _atomic_write_text(out_md, markdown_content)
        logger.info("Wrote %s", out_md)

    # Handle CSV export for todos with basic de-dupe
    if args.write_todos_csv:
        todos = data.get("next_actions") or []
        if todos and isinstance(todos, list):
            root = tdir.parent
            _append_todos_csv(root, tdir.name, todos)


def analyze_conversation_dir_sync(*args, **kwargs):
    """
    Synchronous wrapper for analyze_conversation_dir.
    Runs the async function in an event loop for callers that expect sync API.
    """

    return asyncio.run(analyze_conversation_dir(*args, **kwargs))


def main() -> None:

    asyncio.run(async_main())


if __name__ == "__main__":
    main()
