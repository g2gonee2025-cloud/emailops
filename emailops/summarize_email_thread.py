#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import re
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

try:
    # Attempt relative imports for package context
    from emailops.llm_client import complete_json, complete_text
    from emailops.utils import (
        clean_email_text,
        ensure_dir,
        extract_email_metadata,
        logger,
        read_text_file,
    )
except ImportError:
    # Fallback for script execution: try absolute imports if available
    try:
        from llm_client import complete_json, complete_text
        from utils import (
            clean_email_text,
            ensure_dir,
            extract_email_metadata,
            logger,
            read_text_file,
        )
    except ImportError:
        # Define minimal no-op fallbacks for CLI to run without full package
        import logging
        logger = logging.getLogger(__name__)

        def _no_op_str_passthrough(s: str, *_args, **_kwargs) -> str:
            return s if s else ""

        def _no_op_read_text(p: Path) -> str:
            return p.read_text(encoding="utf-8")

        def _no_op_ensure_dir(p: Path) -> None:
            p.mkdir(parents=True, exist_ok=True)

        def _no_op_extract_email_metadata(_s: str) -> dict:
            return {}

        clean_email_text = _no_op_str_passthrough
        read_text_file = _no_op_read_text
        ensure_dir = _no_op_ensure_dir
        extract_email_metadata = _no_op_extract_email_metadata

        def _raise_runtime_error(*_args, **_kwargs):
            raise RuntimeError(
                "LLM client not found. Please install 'emailops' as a package "
                "or ensure 'llm_client.py' is in your PYTHONPATH."
            )
        complete_json = _raise_runtime_error
        complete_text = _raise_runtime_error

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
    Respects nested braces, strings, and escaped characters.
    Returns the object as a string, or None if no balanced object is found.
    """
    if not s or "{" not in s:
        return None

    first_brace = s.find("{")
    if first_brace == -1:
        return None

    balance = 0
    in_string = False
    is_escaped = False
    for i, char in enumerate(s[first_brace:]):
        if is_escaped:
            is_escaped = False
            continue

        if char == "\\":
            is_escaped = True
            continue

        if char == '"':
            in_string = not in_string

        if in_string:
            continue

        if char == "{":
            balance += 1
        elif char == "}":
            balance -= 1

        if balance == 0:
            return s[first_brace : first_brace + i + 1]

    return None


def _try_load_json(data: Any) -> dict[str, Any] | None:
    """
    Robustly parse JSON from model output, accepting dict, bytes, or string.
    Handles:
      1) Pre-parsed dicts
      2) Byte strings (UTF-8 decoded)
      3) Direct JSON strings
      4) Fenced ```json blocks
      5) First balanced {...} object in the string
    Returns a dict or None if not recoverable.
    """
    if isinstance(data, dict):
        return data
    if not data:
        return None

    s = ""
    if isinstance(data, bytes):
        try:
            s = data.decode("utf-8")
        except UnicodeDecodeError:
            return None  # Not valid UTF-8
    elif isinstance(data, str):
        s = data
    else:
        return None  # Unsupported type

    s = s.strip()
    if not s:
        return None

    # 1) Try direct parsing first
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass  # Continue to extraction methods

    # 2) Look for fenced ```json blocks
    for m in re.finditer(r"```(?:json)?\s*([\s\S]*?)\s*```", s, flags=re.IGNORECASE):
        block = _extract_first_balanced_json_object(m.group(1))
        if block:
            try:
                obj = json.loads(block)
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                continue

    # 3) Fallback to the first balanced object in the whole string
    block = _extract_first_balanced_json_object(s)
    if block:
        try:
            obj = json.loads(block)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass

    return None


def _safe_str(v: Any, max_len: int) -> str:
    s = "" if v is None else str(v)
    if len(s) > max_len:
        s = s[:max_len].rstrip()
    return s


def _md_escape(v: Any) -> str:
    """Escape text for safe Markdown rendering in generated summaries."""
    if v is None:
        return ""
    s = str(v)
    # Escape common markdown-sensitive characters
    return re.sub(r"([\\`*_{}\[\]()#+\-.!>])", r"\\\1", s)


def _limit_list(lst: Any, max_len: int) -> list[Any]:
    if not isinstance(lst, list):
        return []
    if max_len <= 0:
        return []
    return lst[:max_len]


def _normalize_name(n: str) -> str:
    """Normalize a name for de-duplication (simple lower/strip)."""
    if not n:
        return ""
    return re.sub(r"\s+", " ", n.strip().lower())


def _normalize_analysis(data: Any, catalog: list[str]) -> dict[str, Any]:
    """
    Coerce potentially imperfect LLM output to the required schema
    with safe defaults and type checks.
    """
    d: dict[str, Any] = dict(data) if isinstance(data, dict) else {}

    # Required top-level keys
    d.setdefault("category", (catalog or DEFAULT_CATALOG)[-1])
    d.setdefault("subject", "Email thread")
    d.setdefault("participants", [])
    d.setdefault("facts_ledger", {})
    d.setdefault("summary", [])
    d.setdefault("next_actions", [])
    d.setdefault("risk_indicators", [])

    # Validate/normalize category
    if d.get("category") not in (catalog or DEFAULT_CATALOG):
        d["category"] = (catalog or DEFAULT_CATALOG)[-1]

    # Subject length cap per description
    subj = d.get("subject")
    if isinstance(subj, str) and len(subj) > SUBJECT_MAX_LEN:
        d["subject"] = subj[:SUBJECT_MAX_LEN].rstrip()

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
    fl.setdefault("explicit_asks", [])
    fl.setdefault("commitments_made", [])
    fl.setdefault("unknowns", [])
    fl.setdefault("forbidden_promises", [])
    fl.setdefault("key_dates", [])

    # Ensure lists
    for k in (
        "explicit_asks",
        "commitments_made",
        "unknowns",
        "forbidden_promises",
        "key_dates",
    ):
        if not isinstance(fl.get(k), list):
            fl[k] = []
    d["facts_ledger"] = fl

    # Apply size caps defensively
    d["participants"] = _limit_list(d["participants"], MAX_PARTICIPANTS)
    d["summary"] = _limit_list(d["summary"], MAX_SUMMARY_POINTS)
    d["next_actions"] = _limit_list(d["next_actions"], MAX_NEXT_ACTIONS)
    for k in (
        "explicit_asks",
        "commitments_made",
        "unknowns",
        "forbidden_promises",
        "key_dates",
    ):
        fl[k] = _limit_list(fl.get(k, []), MAX_FACT_ITEMS)
    d["facts_ledger"] = fl

    return d


def _read_manifest(convo_dir: Path) -> dict[str, Any]:
    """
    Lightweight manifest reader with BOM tolerance and basic sanitation.
    Returns {} if unavailable or invalid.
    """
    manifest_path = convo_dir / "manifest.json"
    if not manifest_path.exists():
        return {}
    try:
        text = manifest_path.read_text(encoding="utf-8-sig")
        # Minimal sanitation: drop control chars that break JSON
        text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]", "", text)
        return json.loads(text)
    except Exception as e:
        logger.warning("Failed to read manifest at %s: %s", manifest_path, e)
        return {}


def _participants_from_manifest(manifest: dict[str, Any]) -> list[dict[str, str]]:
    """
    Convert first-message participants in manifest to summarizer schema.
    Roles default to 'other'; tone 'neutral'; stance 'N/A' to avoid assumptions.
    """
    out: list[dict[str, str]] = []
    try:
        messages = manifest.get("messages") or []
        first = messages[0] if messages else {}

        def _mk(name: str, email: str, role: str = "other") -> dict[str, str]:
            return {
                "name": _safe_str(name, 80),
                "role": role,
                "email": _safe_str(email, 120),
                "tone": "neutral",
                "stance": "N/A",
            }

        if first.get("from"):
            f = first["from"]
            out.append(_mk(f.get("name", ""), f.get("smtp", ""), role="other"))
        for rec in first.get("to") or []:
            if isinstance(rec, dict):
                out.append(_mk(rec.get("name", ""), rec.get("smtp", ""), role="other"))
        for rec in first.get("cc") or []:
            if isinstance(rec, dict):
                out.append(_mk(rec.get("name", ""), rec.get("smtp", ""), role="other"))
    except Exception:
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
    return _limit_list(deduped, MAX_PARTICIPANTS)


def _merge_manifest_into_analysis(
    analysis: dict[str, Any], convo_dir: Path, raw_thread_text: str
) -> dict[str, Any]:
    """
    Merge subject/participants/dates from manifest + raw headers when the model
    couldn't infer them reliably. Never overrides non-empty, already-populated fields.
    """
    manifest = _read_manifest(convo_dir)

    # Subject: prefer existing; otherwise manifest.smart_subject/subject; otherwise parsed raw headers
    if not (
        isinstance(analysis.get("subject"), str)
        and analysis["subject"].strip()
        and analysis["subject"] != "Email thread"
    ):
        subj_candidates: list[str] = []
        if isinstance(manifest, dict):
            smart = (manifest.get("smart_subject") or "").strip()
            if smart:
                subj_candidates.append(smart)
            subj = (manifest.get("subject") or "").strip()
            if subj:
                subj_candidates.append(subj)
        # Parse from raw headers (before cleaning) using utils helper
        md = extract_email_metadata(raw_thread_text or "")
        if isinstance(md, dict) and md.get("subject"):
            subj_candidates.append(str(md["subject"]).strip())
        # Pick the first non-empty candidate
        if subj_candidates:
            analysis["subject"] = _safe_str(subj_candidates[0], SUBJECT_MAX_LEN)

    # Participants: if model didn't provide any, add from manifest
    if not analysis.get("participants"):
        pts = _participants_from_manifest(manifest) if manifest else []
        if pts:
            analysis["participants"] = pts

    # Key dates: if ledger.key_dates is empty, add start/end
    fl = analysis.get("facts_ledger", {}) or {}
    kd = fl.get("key_dates") or []
    if not kd and isinstance(manifest, dict):
        try:
            time_span = manifest.get("time_span") or {}
            start = time_span.get("start_local") or time_span.get("start")
            end = time_span.get("end_local") or time_span.get("end")
            key_dates: list[dict[str, str]] = []
            if start:
                key_dates.append(
                    {
                        "date": str(start),
                        "event": "Conversation start",
                        "importance": "reference",
                    }
                )
            if end:
                key_dates.append(
                    {
                        "date": str(end),
                        "event": "Conversation end",
                        "importance": "reference",
                    }
                )
            if key_dates:
                fl["key_dates"] = _limit_list(key_dates, MAX_FACT_ITEMS)
                analysis["facts_ledger"] = fl
        except Exception:
            pass

    return analysis


def _retry(callable_fn, *args, retries: int = 2, delay: float = 0.5, **kwargs):
    """
    Tiny helper to retry transient LLM failures with exponential backoff + jitter.
    Exceptions are re-raised after the final attempt.
    """
    attempt = 0
    max_retries = retries if retries is not None else 4
    base_delay = delay if delay is not None else 0.5
    while True:
        try:
            return callable_fn(*args, **kwargs)
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
            time.sleep(sleep_for)


def analyze_email_thread_with_ledger(
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
    if not cleaned_thread.strip():
        now = datetime.now(UTC).isoformat()
        return {
            "category": catalog[-1],
            "subject": "Email thread",
            "participants": [],
            "facts_ledger": {
                "explicit_asks": [],
                "commitments_made": [],
                "unknowns": ["No thread content provided"],
                "forbidden_promises": [],
                "key_dates": [],
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
                    "explicit_asks": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "from": {"type": "string"},
                                "request": {"type": "string"},
                                "urgency": {
                                    "type": "string",
                                    "enum": ["immediate", "high", "medium", "low"],
                                },
                                "status": {
                                    "type": "string",
                                    "enum": [
                                        "pending",
                                        "acknowledged",
                                        "in_progress",
                                        "completed",
                                        "blocked",
                                    ],
                                },
                            },
                            "required": ["from", "request", "urgency", "status"],
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
                    "unknowns": {"type": "array", "items": {"type": "string"}},
                    "forbidden_promises": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
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
                },
                "required": [
                    "explicit_asks",
                    "commitments_made",
                    "unknowns",
                    "forbidden_promises",
                    "key_dates",
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
1. TONE & STANCE: How each participant feels and their position
2. EXPLICIT ASKS: Every request made, by whom, and urgency
3. COMMITMENTS: What was promised, by whom, and feasibility
4. UNKNOWNS: Missing information that affects decisions
5. FORBIDDEN PROMISES: Things we must NOT commit to (e.g., guaranteeing coverage without underwriting)
6. KEY DATES: Critical deadlines and events

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
- Each participant's tone and stance
- All explicit requests and their urgency
- Any commitments or promises made
- Information gaps that need to be filled
- Things we should NOT promise
- Critical dates and deadlines
- Risk indicators

Output valid JSON matching the required schema."""

    # --- Pass 1: Initial analysis (robust JSON parsing) ---
    try:
        _cj_kwargs = {
            "max_output_tokens": 2000,
            "temperature": temperature,
            "response_schema": response_schema,
        }
        if json_stop_sequences:
            _cj_kwargs["stop_sequences"] = json_stop_sequences
        initial_response = _retry(
            complete_json,
            system,
            user,
            **_cj_kwargs,
        )
        parsed = _try_load_json(initial_response)
        if not parsed:
            raise ValueError("Model returned non-parseable JSON")
        initial_analysis = _normalize_analysis(parsed, catalog)
    except Exception as e:
        logger.warning("Failed to get structured analysis: %s", e)
        # Fallback to text mode and try salvage (no stop sequences to avoid truncation)
        _ct_kwargs = {
            "max_output_tokens": 2000,
            "temperature": temperature,
        }
        if text_stop_sequences:
            _ct_kwargs["stop_sequences"] = text_stop_sequences
        fb = _retry(
            complete_text,
            system,
            user,
            **_ct_kwargs,
        )
        parsed_fb = _try_load_json(fb) or {}
        initial_analysis = _normalize_analysis(parsed_fb, catalog)
        if not initial_analysis.get("summary"):
            # Ensure a readable fallback summary if salvage didn't contain one
            initial_analysis["summary"] = [fb[:500]]

        # Ensure unknowns reflect parse failure
        fl = initial_analysis.get("facts_ledger", {})
        if isinstance(fl, dict):
            unknowns = fl.get("unknowns", [])
            if isinstance(unknowns, list):
                unknowns.append("Failed to parse thread properly")
                fl["unknowns"] = unknowns
                initial_analysis["facts_ledger"] = fl

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
            "max_output_tokens": 1000,
            "temperature": 0.1,
            "response_schema": critic_schema,
        }
        if json_stop_sequences:
            _crit_kwargs["stop_sequences"] = json_stop_sequences
        critic_response = _retry(
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
            _imp_kwargs = {
                "max_output_tokens": 2000,
                "temperature": 0.2,
                "response_schema": response_schema,
            }
            if json_stop_sequences:
                _imp_kwargs["stop_sequences"] = json_stop_sequences
            improved_response = _retry(
                complete_json,
                improvement_system,
                improvement_user,
                **_imp_kwargs,
            )
            parsed_imp = _try_load_json(improved_response)
            if parsed_imp:
                final_analysis = _normalize_analysis(parsed_imp, catalog)
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
def analyze_conversation_dir(
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

    data = analyze_email_thread_with_ledger(
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
    
    Args:
        analysis: Analysis dictionary from analyze_email_thread_with_ledger()
        
    Returns:
        Formatted markdown string
    """
    md_parts: list[str] = []
    
    # Header
    md_parts.append("# Email Thread Analysis\n\n")
    
    # Summary section
    md_parts.append("## Summary\n\n")
    md_parts.append(f"**Category**: {analysis.get('category', 'unknown')}  \n")
    md_parts.append(f"**Subject**: {analysis.get('subject', 'No subject')}\n\n")
    
    # Overview
    md_parts.append("### Overview\n\n")
    for point in analysis.get("summary", []):
        md_parts.append(f"- {_md_escape(point)}\n")
    
    # Participants
    md_parts.append("\n## Participants\n\n")
    for p in analysis.get("participants", []):
        if isinstance(p, dict):
            md_parts.append(
                f"- **{_md_escape(p.get('name', ''))}** ({_md_escape(p.get('role', ''))})\n"
                f"  - Email: {_md_escape(p.get('email', ''))}\n"
                f"  - Tone: {_md_escape(p.get('tone', ''))}\n"
                f"  - Stance: {_md_escape(p.get('stance', ''))}\n"
            )
    
    # Facts Ledger
    md_parts.append("\n## Facts Ledger\n\n")
    
    # Explicit Requests
    md_parts.append("### Explicit Requests\n\n")
    for ask in analysis.get("facts_ledger", {}).get("explicit_asks", []):
        if isinstance(ask, dict):
            md_parts.append(
                f"- **From**: {_md_escape(ask.get('from', ''))}\n"
                f"  - **Request**: {_md_escape(ask.get('request', ''))}\n"
                f"  - **Urgency**: {_md_escape(ask.get('urgency', ''))}\n"
                f"  - **Status**: {_md_escape(ask.get('status', ''))}\n\n"
            )
    
    # Commitments Made
    md_parts.append("### Commitments Made\n\n")
    for commit in analysis.get("facts_ledger", {}).get("commitments_made", []):
        if isinstance(commit, dict):
            md_parts.append(
                f"- **By**: {_md_escape(commit.get('by', ''))}\n"
                f"  - **Commitment**: {_md_escape(commit.get('commitment', ''))}\n"
                f"  - **Deadline**: {_md_escape(commit.get('deadline', ''))}\n"
                f"  - **Feasibility**: {_md_escape(commit.get('feasibility', ''))}\n\n"
            )
    
    # Unknown Information
    md_parts.append("### Unknown Information\n\n")
    for unknown in analysis.get("facts_ledger", {}).get("unknowns", []):
        md_parts.append(f"- {_md_escape(unknown)}\n")
    
    # Forbidden Promises
    md_parts.append("\n### Forbidden Promises\n\n")
    for forbidden in analysis.get("facts_ledger", {}).get("forbidden_promises", []):
        md_parts.append(f"- ⚠️ {_md_escape(forbidden)}\n")
    
    # Key Dates
    md_parts.append("\n### Key Dates\n\n")
    for kd in analysis.get("facts_ledger", {}).get("key_dates", []):
        if isinstance(kd, dict):
            md_parts.append(
                f"- **{_md_escape(kd.get('date', ''))}**: {_md_escape(kd.get('event', ''))} "
                f"({_md_escape(kd.get('importance', ''))})\n"
            )
    
    # Next Actions
    md_parts.append("\n## Next Actions\n\n")
    for action in analysis.get("next_actions", []):
        if isinstance(action, dict):
            md_parts.append(
                f"- **{_md_escape(action.get('owner', ''))}**: {_md_escape(action.get('action', ''))}\n"
                f"  - Priority: {_md_escape(action.get('priority', ''))}\n"
                f"  - Status: {_md_escape(action.get('status', ''))}\n"
            )
            if action.get('due'):
                md_parts.append(f"  - Due: {_md_escape(action.get('due', ''))}\n")
            md_parts.append("\n")
    
    # Risk Indicators
    md_parts.append("## Risk Indicators\n\n")
    for risk in analysis.get("risk_indicators", []):
        md_parts.append(f"- 🚨 {_md_escape(risk)}\n")
    
    # Metadata
    if "_metadata" in analysis:
        meta = analysis["_metadata"]
        md_parts.append("\n---\n\n")
        md_parts.append("## Metadata\n\n")
        md_parts.append(f"- **Analyzed at**: {meta.get('analyzed_at', 'N/A')}\n")
        md_parts.append(f"- **Provider**: {meta.get('provider', 'N/A')}\n")
        md_parts.append(f"- **Completeness score**: {meta.get('completeness_score', 0)}%\n")
        md_parts.append(f"- **Version**: {meta.get('version', 'N/A')}\n")
    
    return "".join(md_parts)


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


def _append_todos_csv(
    root: Path, thread_name: str, todos: list[dict[str, Any]]
) -> None:
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
        w = csv.DictWriter(
            f, fieldnames=["who", "what", "due", "status", "priority", "thread"]
        )
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
def main() -> None:
    # Configure logging for CLI entry point
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    ap = argparse.ArgumentParser(
        description="Summarize a single email thread directory with comprehensive facts ledger."
    )
    ap.add_argument(
        "--thread",
        required=True,
        help="Path to a conversation directory containing Conversation.txt",
    )
    ap.add_argument(
        "--catalog", nargs="*", default=DEFAULT_CATALOG, help="Allowed categories"
    )
    ap.add_argument(
        "--write_todos_csv",
        action="store_true",
        help="Append next_actions to root todo.csv",
    )
    ap.add_argument(
        "--provider",
        default=os.getenv("EMBED_PROVIDER", "vertex"),
        help="Provider name recorded in metadata (does not route requests).",
    )
    ap.add_argument(
        "--temperature", type=float, default=0.2, help="Temperature for generation"
    )
    ap.add_argument(
        "--output-format",
        choices=["json", "markdown"],
        default="json",
        help="Write JSON by default; with \"markdown\", also write summary.md. Combine with --no-json to skip JSON.",
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
        data = analyze_conversation_dir(
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

    # Optionally save as markdown for readability
    if args.output_format == "markdown":
        md_parts: list[str] = []
        md_parts.append("# Email Thread Analysis\n")
        md_parts.append("## Summary\n")
        md_parts.append(f"**Category**: {data.get('category', 'unknown')}  \n")
        md_parts.append(f"**Subject**: {data.get('subject', 'No subject')}\n")
        md_parts.append("### Overview\n")
        for point in data.get("summary", []):
            md_parts.append(f"- {_md_escape(point)}\n")

        md_parts.append("\n## Participants\n")
        for p in data.get("participants", []):
            if isinstance(p, dict):
                md_parts.append(
                    f"- **{_md_escape(p.get('name', ''))}** ({_md_escape(p.get('role', ''))})\n  - Tone: {_md_escape(p.get('tone', ''))}\n  - Stance: {_md_escape(p.get('stance', ''))}\n"
                )

        md_parts.append("\n## Facts Ledger\n\n### Explicit Requests\n")
        for ask in data.get("facts_ledger", {}).get("explicit_asks", []):
            if isinstance(ask, dict):
                md_parts.append(
                    f"- **From**: {_md_escape(ask.get('from', ''))}\n"
                    f"  - **Request**: {_md_escape(ask.get('request', ''))}\n"
                    f"  - **Urgency**: {_md_escape(ask.get('urgency', ''))}\n"
                    f"  - **Status**: {_md_escape(ask.get('status', ''))}\n\n"
                )

        md_parts.append("\n### Commitments Made\n")
        for commit in data.get("facts_ledger", {}).get("commitments_made", []):
            if isinstance(commit, dict):
                md_parts.append(
                    f"- **By**: {_md_escape(commit.get('by', ''))}\n"
                    f"  - **Commitment**: {_md_escape(commit.get('commitment', ''))}\n"
                    f"  - **Deadline**: {_md_escape(commit.get('deadline', ''))}\n"
                    f"  - **Feasibility**: {_md_escape(commit.get('feasibility', ''))}\n"
                )

        md_parts.append("\n### Unknown Information\n")
        for unknown in data.get("facts_ledger", {}).get("unknowns", []):
            md_parts.append(f"- {_md_escape(unknown)}\n")

        md_parts.append("\n### Forbidden Promises\n")
        for forbidden in data.get("facts_ledger", {}).get("forbidden_promises", []):
            md_parts.append(f"- ⚠️ {_md_escape(forbidden)}\n")

        md_parts.append("\n## Key Dates\n")
        for kd in data.get("facts_ledger", {}).get("key_dates", []):
            if isinstance(kd, dict):
                md_parts.append(
                    f"- **{_md_escape(kd.get('date', ''))}**: {_md_escape(kd.get('event', ''))} ({_md_escape(kd.get('importance', ''))})\n"
                )

        md_parts.append("\n## Next Actions\n")
        for action in data.get("next_actions", []):
            if isinstance(action, dict):
                md_parts.append(
                    f"- **{_md_escape(action.get('owner', ''))}**: {_md_escape(action.get('action', ''))}\n"
                    f"  - Priority: {_md_escape(action.get('priority', ''))}\n"
                    f"  - Status: {_md_escape(action.get('status', ''))}\n\n"
                )

        md_parts.append("\n## Risk Indicators\n")
        for risk in data.get("risk_indicators", []):
            md_parts.append(f"- 🚨 {_md_escape(risk)}\n")

        out_md = tdir / "summary.md"
        _atomic_write_text(out_md, "".join(md_parts))
        logger.info("Wrote %s", out_md)

    # Handle CSV export for todos with basic de-dupe
    if args.write_todos_csv:
        todos = data.get("next_actions") or []
        if todos and isinstance(todos, list):
            root = tdir.parent
            _append_todos_csv(root, tdir.name, todos)


if __name__ == "__main__":
    main()
