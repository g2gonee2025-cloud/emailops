#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timezone

from .utils import (
    logger,
    clean_email_text,
    read_text_file,
    ensure_dir,
    extract_email_metadata,
)

from .llm_client import complete_json, complete_text

# =============================================================================
# Configuration (env‑overridable) to keep prompts bounded and predictable
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
def _try_load_json(s: str) -> Optional[Dict[str, Any]]:
    """
    Robustly parse JSON from model output:
      1) direct json.loads()
      2) fenced ```json blocks
      3) first {...} span
    Returns dict or None if not recoverable.
    """
    if not s:
        return None
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    fence = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", s, flags=re.IGNORECASE)
    if fence:
        try:
            obj = json.loads(fence.group(1))
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    block = re.search(r"\{[\s\S]*\}", s)
    if block:
        try:
            obj = json.loads(block.group(0))
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    return None


def _safe_str(v: Any, max_len: int) -> str:
    s = "" if v is None else str(v)
    if len(s) > max_len:
        s = s[:max_len].rstrip()
    return s


def _limit_list(lst: Any, max_len: int) -> List[Any]:
    if not isinstance(lst, list):
        return []
    if max_len <= 0:
        return []
    return lst[:max_len]


def _normalize_analysis(data: Any, catalog: List[str]) -> Dict[str, Any]:
    """
    Coerce potentially imperfect LLM output to the required schema
    with safe defaults and type checks.
    """
    d: Dict[str, Any] = dict(data) if isinstance(data, dict) else {}

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
    for k in ("explicit_asks", "commitments_made", "unknowns", "forbidden_promises", "key_dates"):
        if not isinstance(fl.get(k), list):
            fl[k] = []
    d["facts_ledger"] = fl

    # Apply size caps defensively
    d["participants"] = _limit_list(d["participants"], MAX_PARTICIPANTS)
    d["summary"] = _limit_list(d["summary"], MAX_SUMMARY_POINTS)
    d["next_actions"] = _limit_list(d["next_actions"], MAX_NEXT_ACTIONS)
    for k in ("explicit_asks", "commitments_made", "unknowns", "forbidden_promises", "key_dates"):
        fl[k] = _limit_list(fl.get(k, []), MAX_FACT_ITEMS)
    d["facts_ledger"] = fl

    return d


def _read_manifest(convo_dir: Path) -> Dict[str, Any]:
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


def _participants_from_manifest(manifest: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Convert first-message participants in manifest to summarizer schema.
    Roles default to 'other'; tone 'neutral'; stance 'N/A' to avoid assumptions.
    """
    out: List[Dict[str, str]] = []
    try:
        messages = manifest.get("messages") or []
        first = messages[0] if messages else {}
        def _mk(name: str, email: str, role: str = "other") -> Dict[str, str]:
            return {
                "name": _safe_str(name, 80),
                "role": role,
                "email": _safe_str(email, 120),
                "tone": "neutral",
                "stance": "N/A",
            }
        if first.get("from"):
            f = first["from"]
            out.append(_mk(f.get("name",""), f.get("smtp",""), role="other"))
        for rec in (first.get("to") or []):
            if isinstance(rec, dict):
                out.append(_mk(rec.get("name",""), rec.get("smtp",""), role="other"))
        for rec in (first.get("cc") or []):
            if isinstance(rec, dict):
                out.append(_mk(rec.get("name",""), rec.get("smtp",""), role="other"))
    except Exception:
        return out
    # de-duplicate by lowercase email
    seen = set()
    deduped: List[Dict[str, str]] = []
    for p in out:
        key = (p.get("email") or "").lower()
        if key in seen and key:
            continue
        seen.add(key)
        deduped.append(p)
    return _limit_list(deduped, MAX_PARTICIPANTS)


def _merge_manifest_into_analysis(
    analysis: Dict[str, Any],
    convo_dir: Path,
    raw_thread_text: str
) -> Dict[str, Any]:
    """
    Merge subject/participants/dates from manifest + raw headers when the model
    couldn't infer them reliably. Never overrides non-empty, already-populated fields.
    """
    manifest = _read_manifest(convo_dir)

    # Subject: prefer existing; otherwise manifest.smart_subject/subject; otherwise parsed raw headers
    if not (isinstance(analysis.get("subject"), str) and analysis["subject"].strip() and analysis["subject"] != "Email thread"):
        subj_candidates: List[str] = []
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
            key_dates: List[Dict[str, str]] = []
            if start:
                key_dates.append({"date": str(start), "event": "Conversation start", "importance": "reference"})
            if end:
                key_dates.append({"date": str(end), "event": "Conversation end", "importance": "reference"})
            if key_dates:
                fl["key_dates"] = _limit_list(key_dates, MAX_FACT_ITEMS)
                analysis["facts_ledger"] = fl
        except Exception:
            pass

    return analysis


def analyze_email_thread_with_ledger(
    thread_text: str,
    catalog: List[str] = DEFAULT_CATALOG,
    provider: str = "vertex",
    temperature: float = 0.2
) -> Dict[str, Any]:
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
        now = datetime.now(timezone.utc).isoformat()
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
            "category": {"type": "string", "enum": catalog, "description": "Primary category this thread belongs to"},
            "subject": {"type": "string", "description": "Short descriptive title (max 100 chars)"},
            "participants": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "role": {"type": "string", "enum": ["client", "broker", "underwriter", "internal", "other"]},
                        "email": {"type": "string"},
                        "tone": {"type": "string", "enum": ["professional", "frustrated", "urgent", "friendly", "demanding", "neutral"]},
                        "stance": {"type": "string", "description": "Their position/attitude in this thread"},
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
                                "urgency": {"type": "string", "enum": ["immediate", "high", "medium", "low"]},
                                "status": {"type": "string", "enum": ["pending", "acknowledged", "in_progress", "completed", "blocked"]},
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
                                "feasibility": {"type": "string", "enum": ["achievable", "challenging", "risky", "impossible"]},
                            },
                            "required": ["by", "commitment", "feasibility"],
                        },
                    },
                    "unknowns": {"type": "array", "items": {"type": "string"}},
                    "forbidden_promises": {"type": "array", "items": {"type": "string"}},
                    "key_dates": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "date": {"type": "string"},
                                "event": {"type": "string"},
                                "importance": {"type": "string", "enum": ["critical", "important", "reference"]},
                            },
                            "required": ["date", "event", "importance"],
                        },
                    },
                },
                "required": ["explicit_asks", "commitments_made", "unknowns", "forbidden_promises", "key_dates"],
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
                        "status": {"type": "string", "enum": ["open", "in_progress", "blocked", "completed"]},
                        "priority": {"type": "string", "enum": ["critical", "high", "medium", "low"]},
                    },
                    "required": ["owner", "action", "status", "priority"],
                },
            },
            "risk_indicators": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["category", "subject", "participants", "facts_ledger", "summary", "next_actions", "risk_indicators"],
    }

    # Stop sequences (ensure we break on original/forwarded markers)
    stop_sequences = [
        "\n\n---",
        "\n\nFrom:",
        "\n\nSent:",
        "\n\n-----Original Message-----",
        "```",
    ]

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
        initial_response = complete_json(
            system,
            user,
            max_output_tokens=2000,
            temperature=temperature,
            response_schema=response_schema,
            stop_sequences=stop_sequences,
        )
        parsed = _try_load_json(initial_response)
        if not parsed:
            raise ValueError("Model returned non‑parseable JSON")
        initial_analysis = _normalize_analysis(parsed, catalog)
    except Exception as e:
        logger.warning(f"Failed to get structured analysis: {e}")
        # Fallback to text mode and try salvage
        fb = complete_text(
            system,
            user,
            max_output_tokens=2000,
            temperature=temperature,
            stop_sequences=stop_sequences,
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
                    "properties": {"field": {"type": "string"}, "issue": {"type": "string"}, "correction": {"type": "string"}},
                },
            },
            "completeness_score": {"type": "integer", "minimum": 0, "maximum": 100},
            "critical_gaps": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["missed_items", "accuracy_issues", "completeness_score", "critical_gaps"],
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
        critic_response = complete_json(
            critic_system,
            critic_user,
            max_output_tokens=1000,
            temperature=0.1,
            response_schema=critic_schema,
            stop_sequences=stop_sequences,
        )
        critic_feedback = _try_load_json(critic_response) or {
            "missed_items": {},
            "accuracy_issues": [],
            "completeness_score": 80,
            "critical_gaps": [],
        }
    except Exception as e:
        logger.warning(f"Failed to get critic feedback: {e}")
        critic_feedback = {
            "missed_items": {},
            "accuracy_issues": [],
            "completeness_score": 80,
            "critical_gaps": [],
        }

    # --- Pass 3: Improvement if the critic flags gaps ---
    final_analysis = initial_analysis
    if critic_feedback.get("completeness_score", 100) < 85 or critic_feedback.get("critical_gaps"):
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
            improved_response = complete_json(
                improvement_system,
                improvement_user,
                max_output_tokens=2000,
                temperature=0.2,
                response_schema=response_schema,
                stop_sequences=stop_sequences,
            )
            parsed_imp = _try_load_json(improved_response)
            if parsed_imp:
                final_analysis = _normalize_analysis(parsed_imp, catalog)
        except Exception as e:
            logger.warning(f"Failed to get improved analysis: {e}")
            # keep initial_analysis

    # Attach metadata (stable; helpful for auditability)
    final_analysis["_metadata"] = {
        "analyzed_at": datetime.now(timezone.utc).isoformat(),
        "provider": provider,
        "completeness_score": int(critic_feedback.get("completeness_score", 0) or 0),
        "version": ANALYZER_VERSION,
        "input_chars": len(cleaned_thread),
    }
    return final_analysis


# =============================================================================
# Higher‑level API: analyze a conversation directory (read, analyze, enrich)
# =============================================================================
def analyze_conversation_dir(
    thread_dir: Path,
    catalog: List[str] = DEFAULT_CATALOG,
    provider: str = os.getenv("EMBED_PROVIDER", "vertex"),
    temperature: float = 0.2,
    merge_manifest: bool = True,
) -> Dict[str, Any]:
    """
    Read Conversation.txt from `thread_dir`, run the facts‑ledger analysis, and
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


# =============================================================================
# Filesystem utilities
# =============================================================================
def _atomic_write_text(path: Path, content: str, encoding: str = "utf-8") -> None:
    """Write text atomically to avoid partial files on interruptions."""
    ensure_dir(path.parent)
    with tempfile.NamedTemporaryFile("w", delete=False, encoding=encoding, dir=str(path.parent)) as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, path)


def _append_todos_csv(root: Path, thread_name: str, todos: List[Dict[str, Any]]) -> None:
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
            key = (
                str(t.get("owner", "")).strip().lower(),
                str(t.get("action", "")).strip().lower(),
                thread_name.strip().lower(),
            )
            if key in existing_keys:
                continue
            w.writerow(
                {
                    "who": t.get("owner", ""),
                    "what": t.get("action", ""),
                    "due": t.get("due", ""),
                    "status": t.get("status", "open"),
                    "priority": t.get("priority", "medium"),
                    "thread": thread_name,
                }
            )


# =============================================================================
# CLI
# =============================================================================
def main() -> None:
    # Configure logging for CLI entry point
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    ap = argparse.ArgumentParser(description="Summarize a single email thread directory with comprehensive facts ledger.")
    ap.add_argument("--thread", required=True, help="Path to a conversation directory containing Conversation.txt")
    ap.add_argument("--catalog", nargs="*", default=DEFAULT_CATALOG, help="Allowed categories")
    ap.add_argument("--write_todos_csv", action="store_true", help="Append next_actions to root todo.csv")
    ap.add_argument("--provider", default=os.getenv("EMBED_PROVIDER", "vertex"), help="LLM provider to record in metadata")
    ap.add_argument("--temperature", type=float, default=0.2, help="Temperature for generation")
    ap.add_argument("--output-format", choices=["json", "markdown"], default="json", help="Output format")
    ap.add_argument("--no-manifest-merge", action="store_true", help="Do NOT merge manifest metadata (subject/participants/dates)")
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
        raise SystemExit(f"Failed to analyze thread: {e}")

    # Save JSON (atomically)
    out_json = tdir / "summary.json"
    _atomic_write_text(out_json, json.dumps(data, ensure_ascii=False, indent=2))
    logger.info("Wrote %s", out_json)

    # Optionally save as markdown for readability
    if args.output_format == "markdown":
        md_content = f"""# Email Thread Analysis

## Summary
**Category**: {data.get('category', 'unknown')}  
**Subject**: {data.get('subject', 'No subject')}

### Overview
{chr(10).join(f"- {point}" for point in data.get('summary', []))}

## Participants
"""
        for p in data.get("participants", []):
            if isinstance(p, dict):
                md_content += f"\n- **{p.get('name','')}** ({p.get('role','')})\n  - Tone: {p.get('tone','')}\n  - Stance: {p.get('stance','')}\n"

        md_content += "\n## Facts Ledger\n\n### Explicit Requests\n"
        for ask in data.get("facts_ledger", {}).get("explicit_asks", []):
            if isinstance(ask, dict):
                md_content += (
                    f"- **From**: {ask.get('from','')}\n"
                    f"  - **Request**: {ask.get('request','')}\n"
                    f"  - **Urgency**: {ask.get('urgency','')}\n"
                    f"  - **Status**: {ask.get('status','')}\n\n"
                )

        md_content += "\n### Commitments Made\n"
        for commit in data.get("facts_ledger", {}).get("commitments_made", []):
            if isinstance(commit, dict):
                md_content += (
                    f"- **By**: {commit.get('by','')}\n"
                    f"  - **Commitment**: {commit.get('commitment','')}\n"
                    f"  - **Feasibility**: {commit.get('feasibility','')}\n\n"
                )

        md_content += "\n### Unknown Information\n"
        for unknown in data.get("facts_ledger", {}).get("unknowns", []):
            md_content += f"- {unknown}\n"

        md_content += "\n### Forbidden Promises\n"
        for forbidden in data.get("facts_ledger", {}).get("forbidden_promises", []):
            md_content += f"- ⚠️ {forbidden}\n"

        md_content += "\n## Key Dates\n"
        for kd in data.get("facts_ledger", {}).get("key_dates", []):
            if isinstance(kd, dict):
                md_content += f"- **{kd.get('date','')}**: {kd.get('event','')} ({kd.get('importance','')})\n"

        md_content += "\n## Next Actions\n"
        for action in data.get("next_actions", []):
            if isinstance(action, dict):
                md_content += (
                    f"- **{action.get('owner','')}**: {action.get('action','')}\n"
                    f"  - Priority: {action.get('priority','')}\n"
                    f"  - Status: {action.get('status','')}\n\n"
                )

        md_content += "\n## Risk Indicators\n"
        for risk in data.get("risk_indicators", []):
            md_content += f"- 🚨 {risk}\n"

        out_md = tdir / "summary.md"
        _atomic_write_text(out_md, md_content)
        logger.info("Wrote %s", out_md)

    # Handle CSV export for todos with basic de-dupe
    if args.write_todos_csv:
        todos = data.get("next_actions") or []
        if todos and isinstance(todos, list):
            root = tdir.parent
            _append_todos_csv(root, tdir.name, todos)

if __name__ == "__main__":
    main()
