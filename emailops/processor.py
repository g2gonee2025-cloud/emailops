#!/usr/bin/env python3
"""
EmailOps Orchestrator (thin helper)

- One CLI for:
  * index: build/update vector index (delegates to email_indexer.py)
  * reply: draft a reply .eml for a conversation (delegates to search_and_draft.py)
  * fresh: draft a fresh .eml (delegates to search_and_draft.py)
  * chat:  chat with retrieved context (delegates to search_and_draft.py)
  * summarize: summarize one conversation (delegates to summarize_email_thread.py)
  * summarize-many: summarize many conversations in parallel (safe multiprocessing)

- Uses 'spawn' start method and avoids bound-method worker targets.
- Inherits credentials & knobs through config.get_config().update_environment().
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import multiprocessing as mp
import os
import signal
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

# ------------------------------ Imports with robust fallbacks ------------------------------
# Prefer package-relative imports if running inside package; otherwise fall back to sibling files.

try:
    # package form
    from emailops.config import get_config  # type: ignore
    from emailops.index_metadata import INDEX_DIRNAME_DEFAULT  # type: ignore

    # search+draft API
    from emailops.search_and_draft import (
        _search as _low_level_search,  # internal search used for chat context only
    )
    from emailops.search_and_draft import (  # type: ignore
        chat_with_context,
        draft_email_reply_eml,
        draft_fresh_email_eml,
        parse_filter_grammar,
    )

    # summarizer API
    from emailops.summarize_email_thread import (  # type: ignore
        analyze_conversation_dir,
        format_analysis_as_markdown,
    )
except Exception:
    # script form
    from config import get_config  # type: ignore
    from index_metadata import INDEX_DIRNAME_DEFAULT  # type: ignore
    from search_and_draft import _search as _low_level_search
    from search_and_draft import (  # type: ignore
        chat_with_context,
        draft_email_reply_eml,
        draft_fresh_email_eml,
        parse_filter_grammar,
    )
    from summarize_email_thread import (  # type: ignore
        analyze_conversation_dir,
        format_analysis_as_markdown,
    )

# ----------------------------------- Logging -----------------------------------


def _setup_logging(level: str | int = "INFO") -> logging.Logger:
    lvl = getattr(logging, str(level).upper(), logging.INFO)
    logging.basicConfig(
        level=lvl,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )
    return logging.getLogger("emailops.processor")


logger = _setup_logging(os.getenv("LOG_LEVEL", "INFO"))

# ----------------------------------- Custom Exceptions -----------------------------------


class ProcessorError(Exception):
    """Base exception for processor errors."""

    pass


class IndexNotFoundError(ProcessorError):
    """Raised when index directory is not found."""

    pass


class ConfigurationError(ProcessorError):
    """Raised when configuration is invalid."""

    pass


class CommandExecutionError(ProcessorError):
    """Raised when command execution fails."""

    pass


# ----------------------------------- Constants -----------------------------------

# Default timeout for subprocess operations (1 hour)
DEFAULT_SUBPROCESS_TIMEOUT = 3600

# Maximum workers for multiprocessing
MAX_WORKERS_PER_CPU = 0.5  # Use half of available CPUs

# ----------------------------------- Utilities -----------------------------------


def _ensure_env() -> None:
    """
    Ensure environment variables (provider, paths, credentials) are set for
    child processes. Uses the central config singleton.
    """
    cfg = get_config()
    cfg.update_environment()  # sets GOOGLE_APPLICATION_CREDENTIALS, provider, batch, etc.


def _resolve_index_dir(root: Path) -> Path:
    idx_name = os.getenv("INDEX_DIRNAME", INDEX_DIRNAME_DEFAULT)
    return root / idx_name


def _python_module_path(module_name: str) -> Path:
    """
    Resolve a module file path to run via subprocess without depending on package layout.
    """
    try:
        mod = __import__(module_name, fromlist=["__file__"])
        return Path(mod.__file__).resolve()
    except Exception:
        # Fallback: same directory as this script
        return (Path(__file__).parent / f"{module_name.split('.')[-1]}.py").resolve()


def _run_email_indexer(
    *,
    root: Path,
    provider: str = "vertex",
    batch: int | None = None,
    limit: int | None = None,
    force_reindex: bool = False,
    extra_args: list[str] | None = None,
    timeout: int = DEFAULT_SUBPROCESS_TIMEOUT,
) -> int:
    """
    Run the indexer as a *separate process* to keep argparse/sys.argv isolated.
    Now with timeout and better error handling.
    """
    indexer_path = _python_module_path("email_indexer")
    args: list[str] = [sys.executable, str(indexer_path), "--root", str(root), "--provider", provider]
    if batch:
        args += ["--batch", str(batch)]
    if limit:
        args += ["--limit", str(limit)]
    if force_reindex:
        args.append("--force-reindex")
    if extra_args:
        args += list(extra_args)

    # HIGH #14: Enhanced command validation - always validate, never skip
    validate_command_args = None
    # MEDIUM #27: Import path correction - use relative import for validators
    try:
        from .validators import validate_command_args
    except ImportError:
        logger.warning("validators module not available - command validation skipped")

    if validate_command_args is not None:
        ok, msg = validate_command_args(args[0], args[1:])
        if not ok:
            raise CommandExecutionError(f"Unsafe indexer command blocked: {msg}")
    else:
        # HIGH #14: Fallback validation when validators module unavailable
        # Ensure executable is sys.executable (Python interpreter)
        if args[0] != sys.executable:
            raise CommandExecutionError(f"Invalid executable: expected {sys.executable}, got {args[0]}")
        # Basic sanity checks on arguments
        for arg in args[1:]:
            if any(dangerous in str(arg).lower() for dangerous in [";", "&", "|", "`", "$("]):
                raise CommandExecutionError(f"Potentially unsafe argument detected: {arg}")

    env = os.environ.copy()
    _ensure_env()  # ensure config values/creds present

    try:
        proc = subprocess.run(args, env=env, timeout=timeout, capture_output=True, text=True)
        if proc.returncode != 0:
            logger.error("Indexer stderr: %s", proc.stderr)
        return int(proc.returncode or 0)
    except subprocess.TimeoutExpired as e:
        logger.error("Indexer process timed out after %d seconds", timeout)
        raise CommandExecutionError(f"Indexer timed out after {timeout} seconds") from e
    except Exception as e:
        logger.error("Failed to run indexer: %s", e)
        raise CommandExecutionError(f"Failed to run indexer: {e}") from e


# ----------------------------------- Commands -----------------------------------


def cmd_index(ns: argparse.Namespace) -> None:
    root = Path(ns.root).expanduser().resolve()
    if not root.exists():
        raise IndexNotFoundError(f"--root not found: {root}")

    try:
        rc = _run_email_indexer(
            root=root,
            provider=ns.provider,
            batch=ns.batch,
            limit=ns.limit,
            force_reindex=ns.force_reindex,
            extra_args=ns.indexer_args or None,
            timeout=getattr(ns, "timeout", DEFAULT_SUBPROCESS_TIMEOUT),
        )
        if rc != 0:
            raise CommandExecutionError(f"Indexer failed with return code {rc}")
        logger.info("Index build/update complete at %s", _resolve_index_dir(root))
    except ProcessorError:
        raise
    except Exception as e:
        logger.error("Unexpected error in cmd_index: %s", e)
        raise ProcessorError(f"Index command failed: {e}") from e


def cmd_reply(ns: argparse.Namespace) -> None:
    root = Path(ns.root).expanduser().resolve()
    ix_dir = _resolve_index_dir(root)
    if not ix_dir.exists():
        raise IndexNotFoundError(f"Index not found at {ix_dir}. Build it first (see 'index').")

    try:
        result = draft_email_reply_eml(
            export_root=root,
            conv_id=ns.conv_id,
            provider=ns.provider,
            query=ns.query or None,
            sim_threshold=ns.sim_threshold,
            target_tokens=ns.target_tokens,
            temperature=ns.temperature,
            include_attachments=(not ns.no_attachments),
            sender=ns.sender or None,
            reply_to=ns.reply_to or None,
            reply_policy=ns.reply_policy,
        )
        out = ns.out or (root / f"{ns.conv_id}_reply.eml")
        Path(out).write_bytes(result["eml_bytes"])
        print(str(out))
    except Exception as e:
        logger.error("Failed to generate reply: %s", e)
        raise ProcessorError(f"Reply generation failed: {e}") from e


def cmd_fresh(ns: argparse.Namespace) -> None:
    root = Path(ns.root).expanduser().resolve()
    ix_dir = _resolve_index_dir(root)
    if not ix_dir.exists():
        raise IndexNotFoundError(f"Index not found at {ix_dir}. Build it first (see 'index').")

    to_list = [x.strip() for x in (ns.to or "").split(",") if x.strip()]
    if not to_list:
        raise ConfigurationError("--to is required (comma-separated addresses)")
    cc_list = [x.strip() for x in (ns.cc or "").split(",") if x.strip()]
    if not ns.subject:
        raise ConfigurationError("--subject is required")
    if not ns.query:
        raise ConfigurationError("--query is required")

    # Best-effort inline grammar parse for filters (from search_and_draft)
    # The function is consumed internally by draft_fresh_email_eml, but parsing early helps UX.
    try:
        _ = parse_filter_grammar(ns.query)
    except Exception as e:
        raise ConfigurationError(f"Invalid query grammar: {e}") from e

    try:
        result = draft_fresh_email_eml(
            export_root=root,
            provider=ns.provider,
            to_list=to_list,
            cc_list=cc_list,
            subject=ns.subject,
            query=ns.query,
            sim_threshold=ns.sim_threshold,
            target_tokens=ns.target_tokens,
            temperature=ns.temperature,
            include_attachments=(not ns.no_attachments),
            sender=ns.sender or None,
            reply_to=ns.reply_to or None,
        )
        out = ns.out or (root / f"fresh_{Path(ns.subject).stem}.eml")
        Path(out).write_bytes(result["eml_bytes"])
        print(str(out))
    except Exception as e:
        logger.error("Failed to generate fresh email: %s", e)
        raise ProcessorError(f"Fresh email generation failed: {e}") from e


def cmd_chat(ns: argparse.Namespace) -> None:
    root = Path(ns.root).expanduser().resolve()
    ix_dir = _resolve_index_dir(root)
    if not ix_dir.exists():
        raise IndexNotFoundError(f"Index not found at {ix_dir}. Build it first (see 'index').")
    if not ns.query:
        raise ConfigurationError("--query is required for chat")

    try:
        # Gather context using the same internal search your drafting uses.
        ctx = _low_level_search(ix_dir=ix_dir, query=ns.query, k=ns.k, provider=ns.provider)
        # Set provider in environment for chat_with_context
        os.environ["EMBED_PROVIDER"] = ns.provider
        data = chat_with_context(
            context_snippets=ctx,
            query=ns.query,
            temperature=ns.temperature,
            chat_history=None,  # can be wired to a persisted session later if desired
        )
        if ns.json:
            print(json.dumps(data, ensure_ascii=False, indent=2))
        else:
            print(data.get("answer", "").strip())
    except Exception as e:
        logger.error("Chat failed: %s", e)
        raise ProcessorError(f"Chat command failed: {e}") from e


# ----------------------------------- Public API for GUI and external callers -----------------------------------


def _search(*args, **kwargs):
    """
    Public wrapper for search functionality (used by GUI).
    Delegates to internal search implementation.
    """
    return _low_level_search(*args, **kwargs)


# Re-export for GUI compatibility (already imported above)
# list_conversations_newest_first is imported from search_and_draft and available


# ----------------------- Summarize one / many (multiprocessing) -----------------------


@dataclass(frozen=True)
class _SummJob:
    convo_dir: Path
    provider: str
    out_dir: Path | None


def _summarize_worker(job: _SummJob) -> tuple[str, bool, str]:
    """
    Top-level picklable worker (safe under 'spawn').
    Returns: (convo_dir, success, message)
    """
    import asyncio

    try:
        _ensure_env()
        # Set provider in environment for analyze_conversation_dir
        os.environ["EMBED_PROVIDER"] = job.provider
        analysis = asyncio.run(analyze_conversation_dir(job.convo_dir))
        md = format_analysis_as_markdown(analysis)
        target_dir = job.out_dir or job.convo_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        (target_dir / "analysis.json").write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")
        (target_dir / "analysis.md").write_text(md, encoding="utf-8")
        return (str(job.convo_dir), True, "ok")
    except Exception as e:
        return (str(job.convo_dir), False, f"{e}")


def _iter_conversation_dirs(root: Path) -> Iterable[Path]:
    # A conversation dir is any subfolder containing a Conversation.txt (your summarizer expects that)
    for p in root.iterdir():
        if p.is_dir() and (p / "Conversation.txt").exists():
            yield p


def cmd_summarize(ns: argparse.Namespace) -> None:
    import asyncio

    convo_dir = Path(ns.conversation).expanduser().resolve()
    if not convo_dir.exists():
        raise IndexNotFoundError(f"Conversation directory not found: {convo_dir}")

    try:
        _ensure_env()
        # analyze_conversation_dir doesn't have a provider param - it gets it from env
        # Set the environment variable before calling
        os.environ["EMBED_PROVIDER"] = ns.provider
        analysis = asyncio.run(analyze_conversation_dir(convo_dir))
        md = format_analysis_as_markdown(analysis)
        out_dir = Path(ns.out).expanduser().resolve() if ns.out else convo_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "analysis.json").write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")
        (out_dir / "analysis.md").write_text(md, encoding="utf-8")
        print(str(out_dir / "analysis.json"))
    except Exception as e:
        logger.error("Summarization failed: %s", e)
        raise ProcessorError(f"Summarization failed: {e}") from e


def cmd_summarize_many(ns: argparse.Namespace) -> None:
    root = Path(ns.root).expanduser().resolve()
    if not root.exists():
        raise IndexNotFoundError(f"--root not found: {root}")

    # Fix: Provide out_dir correctly based on ns.out
    out_dir = Path(ns.out) if ns.out else None
    jobs = [_SummJob(convo_dir=d, provider=ns.provider, out_dir=out_dir) for d in _iter_conversation_dirs(root)]
    if not jobs:
        print("No conversation folders found (need subfolder with Conversation.txt)")
        return

    # Safe multiprocessing: spawn, simple top-level worker, graceful shutdown
    ctx = mp.get_context("spawn")
    success = 0

    # Limit workers to avoid resource exhaustion
    max_workers = min(ns.workers, int((os.cpu_count() or 2) * MAX_WORKERS_PER_CPU))

    try:
        with ctx.Pool(processes=max(1, max_workers), maxtasksperchild=10) as pool:
            for convo_dir, ok, msg in pool.imap_unordered(_summarize_worker, jobs):
                if ok:
                    success += 1
                    logger.info("Summarized: %s", convo_dir)
                else:
                    logger.warning("Failed: %s -> %s", convo_dir, msg)
    except KeyboardInterrupt:
        logger.warning("Interrupted. Workers will terminate.")
    except Exception as e:
        logger.error("Multiprocessing error: %s", e)
        raise ProcessorError(f"Batch summarization failed: {e}") from e

    print(f"Summarized {success}/{len(jobs)} conversations")


# ----------------------------------- CLI -----------------------------------


def build_cli() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="EmailOps processor (thin orchestrator)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # index
    p = sub.add_parser("index", help="Build or update the vector index")
    p.add_argument("--root", required=True, help="Export root containing conversations")
    p.add_argument("--provider", default=os.getenv("EMBED_PROVIDER", "vertex"))
    p.add_argument("--batch", type=int, default=int(os.getenv("EMBED_BATCH", "64")))
    p.add_argument("--limit", type=int, help="Debug: max docs to embed")
    p.add_argument("--force-reindex", action="store_true")
    p.add_argument("--indexer-args", nargs=argparse.REMAINDER, help="Pass-through args to email_indexer")
    p.set_defaults(func=cmd_index)

    # reply
    r = sub.add_parser("reply", help="Draft a reply .eml for a conversation")
    r.add_argument("--root", required=True)
    r.add_argument("--conv-id", required=True)
    r.add_argument("--query", help="Optional query; default is derived from last inbound")
    r.add_argument("--provider", default=os.getenv("EMBED_PROVIDER", "vertex"))
    r.add_argument("--sim-threshold", type=float, default=float(os.getenv("SIM_THRESHOLD_DEFAULT", "0.30")))
    r.add_argument("--target-tokens", type=int, default=int(os.getenv("REPLY_TOKENS_TARGET_DEFAULT", "20000")))
    r.add_argument("--temperature", type=float, default=0.2)
    r.add_argument("--no-attachments", action="store_true")
    r.add_argument("--sender", help='Override sender (must be allow-listed), e.g., "Jane <jane@domain>"')
    r.add_argument("--reply-to", help="Optional Reply-To")
    r.add_argument(
        "--reply-policy", choices=["reply_all", "smart", "sender_only"], default=os.getenv("REPLY_POLICY", "reply_all")
    )
    r.add_argument("--out", help="Output .eml path")
    r.set_defaults(func=cmd_reply)

    # fresh
    f = sub.add_parser("fresh", help="Draft a fresh .eml addressed to provided recipients")
    f.add_argument("--root", required=True)
    f.add_argument("--provider", default=os.getenv("EMBED_PROVIDER", "vertex"))
    f.add_argument("--to", required=True, help="Comma-separated To")
    f.add_argument("--cc", help="Comma-separated Cc")
    f.add_argument("--subject", required=True)
    f.add_argument("--query", required=True, help="Intent/instructions; supports inline filters")
    f.add_argument("--sim-threshold", type=float, default=float(os.getenv("SIM_THRESHOLD_DEFAULT", "0.30")))
    f.add_argument("--target-tokens", type=int, default=int(os.getenv("FRESH_TOKENS_TARGET_DEFAULT", "10000")))
    f.add_argument("--temperature", type=float, default=0.2)
    f.add_argument("--no-attachments", action="store_true")
    f.add_argument("--sender", help="Override sender (allow-list enforced by your downstream)")
    f.add_argument("--reply-to", help="Optional Reply-To")
    f.add_argument("--out", help="Output .eml path")
    f.set_defaults(func=cmd_fresh)

    # chat
    c = sub.add_parser("chat", help="Chat grounded in retrieved context")
    c.add_argument("--root", required=True)
    c.add_argument("--query", required=True)
    c.add_argument("--k", type=int, default=12)
    c.add_argument("--provider", default=os.getenv("EMBED_PROVIDER", "vertex"))
    c.add_argument("--temperature", type=float, default=0.2)
    c.add_argument("--json", action="store_true", help="Emit raw JSON")
    c.set_defaults(func=cmd_chat)

    # summarize one
    s = sub.add_parser("summarize", help="Summarize a single conversation directory")
    s.add_argument("--conversation", required=True, help="Path to conversation folder (must contain Conversation.txt)")
    s.add_argument("--provider", default=os.getenv("EMBED_PROVIDER", "vertex"))
    s.add_argument("--out", help="Optional output directory (defaults to the conversation dir)")
    s.set_defaults(func=cmd_summarize)

    # summarize many
    m = sub.add_parser("summarize-many", help="Summarize all conversations under a root (parallel)")
    m.add_argument("--root", required=True)
    m.add_argument("--provider", default=os.getenv("EMBED_PROVIDER", "vertex"))
    m.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    m.add_argument("--out", help="Optional directory to write summaries (defaults to each conversation dir)")
    m.set_defaults(func=cmd_summarize_many)

    return ap


def main() -> int:
    # Safety: always use spawn (works on Windows + prevents state leakage)
    with contextlib.suppress(RuntimeError):
        mp.set_start_method("spawn", force=True)

    _ensure_env()
    ap = build_cli()
    ns = ap.parse_args()
    # Graceful Ctrl+C
    signal.signal(signal.SIGINT, signal.default_int_handler)

    try:
        ns.func(ns)
        return 0
    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        return 130
    except IndexNotFoundError as e:
        logger.error("Index error: %s", e)
        return 2
    except ConfigurationError as e:
        logger.error("Configuration error: %s", e)
        return 3
    except CommandExecutionError as e:
        logger.error("Command execution error: %s", e)
        return 4
    except ProcessorError as e:
        logger.error("Processor error: %s", e)
        return 5
    except SystemExit:
        raise
    except Exception as e:
        logger.exception("Unexpected error: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
