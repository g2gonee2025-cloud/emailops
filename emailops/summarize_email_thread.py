#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone

from .utils import logger, clean_email_text, read_text_file, ensure_dir
from .llm_client import complete_json, complete_text

# -----------------------------------------------------------------------------
# Configuration (env-overridable) to keep prompts bounded and predictable
# -----------------------------------------------------------------------------
MAX_THREAD_CHARS = int(os.getenv("SUMMARIZER_THREAD_MAX_CHARS", "16000"))
CRITIC_THREAD_CHARS = int(os.getenv("SUMMARIZER_CRITIC_MAX_CHARS", "5000"))
IMPROVE_THREAD_CHARS = int(os.getenv("SUMMARIZER_IMPROVE_MAX_CHARS", "8000"))

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

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
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


def _normalize_analysis(data: Any, catalog: List[str]) -> Dict[str, Any]:
    """
    Coerce potentially imperfect LLM output to the required schema
    with safe defaults and type checks.
    """
    d: Dict[str, Any] = dict(data) if isinstance(data, dict) else {}

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

    # Subject length cap per description
    subj = d.get("subject")
    if isinstance(subj, str) and len(subj) > 100:
        d["subject"] = subj[:100].rstrip()

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

    return d


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
    # Sanitize defensively (callers outside CLI may pass raw text)
    cleaned_thread = clean_email_text(thread_text or "")
    if not cleaned_thread.strip():
        # Return a minimal, schema-compliant object
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
                "version": "2.1-facts-ledger",
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
        "version": "2.1-facts-ledger",
        "input_chars": len(cleaned_thread),
    }
    return final_analysis


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
    args = ap.parse_args()

    tdir = Path(args.thread).expanduser().resolve()
    convo = tdir / "Conversation.txt"
    if not convo.exists():
        raise SystemExit(f"Conversation.txt not found in {tdir}")

    raw = read_text_file(convo)
    cleaned = clean_email_text(raw)

    # Enhanced analysis
    data = analyze_email_thread_with_ledger(
        thread_text=cleaned,
        catalog=args.catalog,
        provider=args.provider,
        temperature=args.temperature,
    )

    # Save as JSON
    out_json = tdir / "summary.json"
    out_json.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
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
        out_md.write_text(md_content, encoding="utf-8")
        logger.info("Wrote %s", out_md)

    # Handle CSV export for todos
    if args.write_todos_csv:
        todos = data.get("next_actions") or []
        if todos and isinstance(todos, list):
            root = tdir.parent
            out = root / "todo.csv"
            exists = out.exists()
            with out.open("a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=["who", "what", "due", "status", "priority", "thread"])
                if not exists:
                    w.writeheader()
                for t in todos:
                    if isinstance(t, dict):
                        w.writerow(
                            {
                                "who": t.get("owner", ""),
                                "what": t.get("action", ""),
                                "due": t.get("due", ""),
                                "status": t.get("status", "open"),
                                "priority": t.get("priority", "medium"),
                                "thread": str(tdir.name),
                            }
                        )
            logger.info("Appended todos to %s", out)


if __name__ == "__main__":
    main()
