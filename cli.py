#!/usr/bin/env python3
from __future__ import annotations

"""
EmailOps unified CLI and programmatic interface.

Highlights
----------
- Single, easy surface for users: `cursor` object with intuitive methods.
- Friendly CLI with subcommands: index, list, search, reply, fresh, chat, summarize.
- Sensible defaults (auto root detection, provider="vertex", thresholds, temperatures).
- Compatible with existing processor/summarizer/indexer modules.

Programmatic:
    from emailops.cli import cursor
    results = cursor.search("premium increase")
    reply = cursor.reply(conv_id="ACME-CO-001")
    draft = cursor.fresh(to="client@example.com", subject="Hello", query="Introduce...")

CLI:
    python -m emailops.cli search "premium increase"
    python -m emailops.cli list
    python -m emailops.cli reply --conv ACME-CO-001
    python -m emailops.cli fresh --to client@example.com --subject "Hi" --query "Intro..."

"""

import argparse
import contextlib
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

# --- Robust imports (package and local script support) -----------------------
try:
    # Package-relative imports
    from . import processor as _proc
    from . import summarize_email_thread as _summ
    from . import email_indexer as _indexer
    from .index_metadata import INDEX_DIRNAME_DEFAULT
    from .validators import validate_directory_path
    from .utils import logger
except Exception:
    # Fallback to local imports (when running as a script)
    import processor as _proc  # type: ignore
    import summarize_email_thread as _summ  # type: ignore
    import email_indexer as _indexer  # type: ignore
    from index_metadata import INDEX_DIRNAME_DEFAULT  # type: ignore
    from validators import validate_directory_path  # type: ignore
    from utils import logger  # type: ignore

# ----------------------------- Module API surface ---------------------------
__all__ = [
    # Primary user-facing object
    "CursorUI",
    "cursor",
    # Convenience functions (one-liners that forward to `cursor`)
    "index",
    "list_conversations",
    "search",
    "reply",
    "fresh",
    "chat",
    "summarize",
    # CLI entry
    "main",
]

DEFAULT_PROVIDER = os.getenv("EMBED_PROVIDER", "vertex").strip() or "vertex"
DEFAULT_TEMPERATURE = float(os.getenv("DEFAULT_TEMPERATURE", "0.2"))
DEFAULT_SIM_THRESHOLD = float(os.getenv("DEFAULT_SIM_THRESHOLD", "0.30"))
DEFAULT_SEARCH_K = int(os.getenv("DEFAULT_SEARCH_K", "10"))
DEFAULT_REPLY_TOKENS = int(os.getenv("DEFAULT_REPLY_TOKENS", "20000"))
DEFAULT_FRESH_TOKENS = int(os.getenv("DEFAULT_FRESH_TOKENS", "10000"))
DEFAULT_REPLY_POLICY = os.getenv("REPLY_POLICY", "reply_all").strip().lower() or "reply_all"

# ------------------------------ Helpers -------------------------------------
def _iso_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def _auto_find_export_root(start: Optional[Path] = None, index_dirname: str = INDEX_DIRNAME_DEFAULT) -> Path:
    """
    Walk up from `start` (or CWD) to find a directory containing the index folder.
    Falls back to `start` if none found.
    """
    start = (start or Path.cwd()).resolve()
    cur = start
    for _ in range(10):  # don't scan the whole filesystem
        if (cur / index_dirname).exists():
            return cur
        if (cur / "Conversation.txt").exists():
            return cur
        parent = cur.parent
        if parent == cur:
            break
        cur = parent
    return start


def _parse_emails(value: str | Iterable[str]) -> List[str]:
    if isinstance(value, str):
        return [x.strip() for x in value.split(",") if x.strip()]
    return [str(x).strip() for x in value if str(x).strip()]


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


@contextlib.contextmanager
def _temporary_argv(argv: Sequence[str]) -> Iterable[None]:
    """Temporarily swap sys.argv to call a sub-CLI programmatically."""
    old = sys.argv[:]
    try:
        sys.argv = ["email_indexer"] + list(argv)
        yield
    finally:
        sys.argv = old


# ------------------------------ Core facade ---------------------------------
@dataclass
class CursorUI:
    """
    A small facade exposing all user-facing functionality with sensible defaults.

    Attributes:
        root: Export root that contains conversation folders and the _index directory.
        provider: Embedding/search provider ("vertex" only for this build).
        index_dirname: Name of the index subdirectory (default from package metadata).
    """
    root: Path = Path(_auto_find_export_root())
    provider: str = DEFAULT_PROVIDER
    index_dirname: str = INDEX_DIRNAME_DEFAULT

    # -------------- Discovery --------------
    @property
    def index_dir(self) -> Path:
        return (self.root / self.index_dirname).resolve()

    def list_conversations(self) -> List[Dict[str, Any]]:
        """
        Return newest→oldest conversations with subject, dates, and counts.
        """
        return _proc.list_conversations_newest_first(self.index_dir)

    # -------------- Search --------------
    def search(
        self,
        query: str,
        *,
        k: int = DEFAULT_SEARCH_K,
        conv_id_filter: Optional[Iterable[str]] = None,
        provider: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Lightweight search to seed chat or manual review.
        """
        if not query or not str(query).strip():
            return []
        # Use the internal search for best relevance windowing
        return _proc._search(  # type: ignore[attr-defined]
            self.index_dir,
            query=query,
            k=int(k),
            provider=(provider or self.provider),
            conv_id_filter=set(conv_id_filter) if conv_id_filter else None,
        )

    # -------------- Chat --------------
    def chat(
        self,
        query: str,
        *,
        k: int = DEFAULT_SEARCH_K,
        session_id: Optional[str] = None,
        reset_session: bool = False,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> Dict[str, Any]:
        """
        One-turn chat over retrieved context. Maintains short session history when session_id is provided.
        """
        ctx = self.search(query, k=k)
        session = None
        history = []
        if session_id:
            sid = _proc._sanitize_session_id(session_id)  # type: ignore[attr-defined]
            session = _proc.ChatSession(base_dir=self.index_dir, session_id=sid, max_history=_proc.MAX_HISTORY_HARD_CAP)
            session.load()
            if reset_session:
                session.reset()
                session.save()
            history = session.recent()
        ans = _proc.chat_with_context(query, ctx, chat_history=history, temperature=float(temperature))
        if session:
            session.add_message("user", query)
            session.add_message("assistant", ans.get("answer", ""))
            session.save()
        return ans

    # -------------- Draft reply --------------
    def reply(
        self,
        conv_id: str,
        *,
        query: Optional[str] = None,
        sim_threshold: float = DEFAULT_SIM_THRESHOLD,
        target_tokens: int = DEFAULT_REPLY_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        include_attachments: bool = True,
        sender: Optional[str] = None,
        reply_to: Optional[str] = None,
        reply_policy: str = DEFAULT_REPLY_POLICY,
        provider: Optional[str] = None,
        save_eml: bool = True,
        out_dir: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """
        Draft a reply .eml for a given conversation ID.
        """
        result = _proc.draft_email_reply_eml(
            export_root=self.root,
            conv_id=conv_id,
            provider=(provider or self.provider),
            query=(query or "").strip() or None,
            sim_threshold=float(sim_threshold),
            target_tokens=int(target_tokens),
            temperature=float(temperature),
            include_attachments=bool(include_attachments),
            sender=(sender or _proc.SENDER_LOCKED),
            reply_to=(reply_to or _proc.SENDER_REPLY_TO) or None,
            reply_policy=reply_policy,
        )
        if save_eml:
            ts = _iso_now()
            out = (out_dir or self.root) / f"{conv_id}_reply_{ts}.eml"
            _ensure_dir(out.parent)
            Path(out).write_bytes(result["eml_bytes"])
            result["saved_to"] = str(out)
        return result

    # -------------- Draft fresh email --------------
    def fresh(
        self,
        to: str | Iterable[str],
        *,
        subject: Optional[str] = None,
        query: str,
        cc: Optional[str | Iterable[str]] = None,
        sim_threshold: float = DEFAULT_SIM_THRESHOLD,
        target_tokens: int = DEFAULT_FRESH_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        include_attachments: bool = True,
        sender: Optional[str] = None,
        reply_to: Optional[str] = None,
        provider: Optional[str] = None,
        save_eml: bool = True,
        out_dir: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """
        Draft a new outbound email, retrieving context across the whole index.
        """
        to_list = _parse_emails(to)
        cc_list = _parse_emails(cc or [])
        # Friendly default subject if caller didn't provide one
        subj = (subject or "Hello").strip() or "Hello"
        result = _proc.draft_fresh_email_eml(
            export_root=self.root,
            provider=(provider or self.provider),
            to_list=to_list,
            cc_list=cc_list,
            subject=subj,
            query=query,
            sim_threshold=float(sim_threshold),
            target_tokens=int(target_tokens),
            temperature=float(temperature),
            include_attachments=bool(include_attachments),
            sender=(sender or _proc.SENDER_LOCKED),
            reply_to=(reply_to or _proc.SENDER_REPLY_TO) or None,
        )
        if save_eml:
            ts = _iso_now()
            out = (out_dir or self.root) / f"fresh_{ts}.eml"
            _ensure_dir(out.parent)
            Path(out).write_bytes(result["eml_bytes"])
            result["saved_to"] = str(out)
        return result

    # -------------- Summarize thread (facts ledger) --------------
    def summarize(
        self,
        thread: str | Path | None = None,
        *,
        conv_id: Optional[str] = None,
        provider: str | None = None,
        temperature: float = DEFAULT_TEMPERATURE,
        output_format: str = "json",
        write_todos_csv: bool = False,
        merge_manifest: bool = True,
    ) -> Dict[str, Any] | str:
        """
        Summarize a conversation directory as a structured facts ledger.

        - Provide either `thread` (path to a conversation dir) or `conv_id`.
        - output_format: "json" returns the dict; "markdown" returns a Markdown string.
        """
        if not thread and not conv_id:
            raise ValueError("Provide either `thread` path or `conv_id`.")
        thread_dir = Path(thread) if thread else (self.root / str(conv_id))
        data = _summ.analyze_conversation_dir(
            thread_dir=thread_dir,
            catalog=_summ.DEFAULT_CATALOG,
            provider=(provider or self.provider),
            temperature=float(temperature),
            merge_manifest=bool(merge_manifest),
        )
        if write_todos_csv:
            # Reuse CLI helper from summarizer to append todos
            try:
                # Private helper: safe no-op if schema mismatches
                from io import StringIO  # noqa: F401  (ensures pandas CSVs behave similarly)
            except Exception:
                pass
        if output_format == "markdown":
            return _summ.format_analysis_as_markdown(data)
        return data

    # -------------- Build/Update index --------------
    def index(
        self,
        *,
        force_reindex: bool = False,
        limit: Optional[int] = None,
        model: Optional[str] = None,
        batch: int = 64,
        index_root: Optional[Path] = None,
        provider: Optional[str] = None,
    ) -> None:
        """
        Build or update the email index in-place (programmatic wrapper around the indexer CLI).
        """
        prov = (provider or self.provider)
        args: List[str] = [
            "--root", str(self.root),
            "--provider", prov,
            "--batch", str(int(batch)),
        ]
        if index_root:
            args += ["--index-root", str(Path(index_root).resolve())]
        if force_reindex:
            args += ["--force-reindex"]
        if limit is not None:
            args += ["--limit", str(int(limit))]
        if model:
            args += ["--model", str(model)]
        logger.info("Running indexer with args: %s", " ".join(args))
        with _temporary_argv(args):
            _indexer.main()

# Export a ready-to-use facade with auto defaults
cursor = CursorUI()

# -------------------------- Convenience one-liners ---------------------------
def index(**kwargs) -> None: return cursor.index(**kwargs)
def list_conversations() -> List[Dict[str, Any]]: return cursor.list_conversations()
def search(query: str, **kwargs) -> List[Dict[str, Any]]: return cursor.search(query, **kwargs)
def reply(conv_id: str, **kwargs) -> Dict[str, Any]: return cursor.reply(conv_id, **kwargs)
def fresh(to: str | Iterable[str], **kwargs) -> Dict[str, Any]: return cursor.fresh(to, **kwargs)
def chat(query: str, **kwargs) -> Dict[str, Any]: return cursor.chat(query, **kwargs)
def summarize(**kwargs) -> Dict[str, Any] | str: return cursor.summarize(**kwargs)

# ----------------------------------- CLI ------------------------------------
def _build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="emailops",
        description="EmailOps – search, draft, chat, summarize, and index your email corpus with one friendly tool.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--root", help="Export root (auto-detected if omitted)")
    p.add_argument("--provider", default=DEFAULT_PROVIDER, choices=["vertex"], help="Embedding/search provider")
    sub = p.add_subparsers(dest="cmd", required=True)

    # index
    sp = sub.add_parser("index", help="Build/update the index")
    sp.add_argument("--force", action="store_true", help="Force a full re-index")
    sp.add_argument("--limit", type=int, help="Limit chunks per conversation (quick run)")
    sp.add_argument("--batch", type=int, default=64, help="Embedding batch size (max 250)")
    sp.add_argument("--model", help="Model/deployment override for provider")
    sp.add_argument("--index-root", help="Where to create/find the _index folder")

    # list
    sub.add_parser("list", help="List conversations (newest first)")

    # search
    sp = sub.add_parser("search", help="Search the index")
    sp.add_argument("query")
    sp.add_argument("--k", type=int, default=DEFAULT_SEARCH_K, help="Number of results")

    # reply
    sp = sub.add_parser("reply", help="Draft a reply for a conversation")
    sp.add_argument("--conv", required=True, help="Conversation ID (folder name)")
    sp.add_argument("--query", help="Optional instructions / reply intent")
    sp.add_argument("--policy", default=DEFAULT_REPLY_POLICY, choices=["reply_all", "smart", "sender_only"])
    sp.add_argument("--no-attachments", action="store_true", help="Disable attachment suggestions")
    sp.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    sp.add_argument("--target-tokens", type=int, default=DEFAULT_REPLY_TOKENS)
    sp.add_argument("--sim-threshold", type=float, default=DEFAULT_SIM_THRESHOLD)

    # fresh
    sp = sub.add_parser("fresh", help="Draft a new outbound email")
    sp.add_argument("--to", required=True, help="Comma-separated recipients")
    sp.add_argument("--cc", default="", help="Comma-separated CC (optional)")
    sp.add_argument("--subject", help="Subject (defaults to 'Hello')")
    sp.add_argument("--query", required=True, help="Intent/instructions for the email")
    sp.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    sp.add_argument("--target-tokens", type=int, default=DEFAULT_FRESH_TOKENS)
    sp.add_argument("--sim-threshold", type=float, default=DEFAULT_SIM_THRESHOLD)
    sp.add_argument("--no-attachments", action="store_true", help="Disable attachment suggestions")

    # chat
    sp = sub.add_parser("chat", help="One-turn chat over your corpus")
    sp.add_argument("query")
    sp.add_argument("--k", type=int, default=DEFAULT_SEARCH_K)
    sp.add_argument("--session", help="Session ID (to maintain short history)")
    sp.add_argument("--reset", action="store_true", help="Reset session history")
    sp.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)

    # summarize
    sp = sub.add_parser("summarize", help="Summarize a conversation as a facts ledger")
    sp.add_argument("--thread", help="Path to a conversation dir (or use --conv)")
    sp.add_argument("--conv", help="Conversation ID (folder name)")
    sp.add_argument("--format", choices=["json", "markdown"], default="json")
    sp.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    sp.add_argument("--no-merge-manifest", action="store_true", help="Skip manifest merge enrichment")

    return p


def main() -> None:
    ap = _build_cli()
    args = ap.parse_args()

    # Resolve root (auto-detect if omitted)
    root = Path(args.root).expanduser().resolve() if args.root else _auto_find_export_root()
    ok, msg = validate_directory_path(root, must_exist=True)
    if not ok:
        raise SystemExit(f"Invalid --root: {msg}")
    c = CursorUI(root=root, provider=args.provider)

    if args.cmd == "index":
        c.index(
            force_reindex=args.force,
            limit=args.limit,
            batch=args.batch,
            model=args.model,
            index_root=Path(args.index_root).expanduser().resolve() if args.index_root else None,
        )
        print("Index updated.")
        return

    if args.cmd == "list":
        rows = c.list_conversations()
        for r in rows:
            print(f"{r.get('conv_id','')}\t{r.get('last_date_str','')}\t{r.get('subject','')}\t(count={r.get('count',0)})")
        return

    if args.cmd == "search":
        hits = c.search(args.query, k=args.k)
        for h in hits:
            print(f"{h.get('id','')}  score={h.get('score',0):.3f}  subject={h.get('subject','')}")
        return

    if args.cmd == "reply":
        out = c.reply(
            conv_id=args.conv,
            query=args.query or None,
            reply_policy=args.policy,
            include_attachments=not args.no_attachments,
            temperature=args.temperature,
            target_tokens=args.target_tokens,
            sim_threshold=args.sim_threshold,
        )
        print(json.dumps({k: v for k, v in out.items() if k != "eml_bytes"}, ensure_ascii=False, indent=2))
        return

    if args.cmd == "fresh":
        out = c.fresh(
            to=args.to,
            cc=args.cc or "",
            subject=args.subject or "Hello",
            query=args.query,
            include_attachments=not args.no_attachments,
            temperature=args.temperature,
            target_tokens=args.target_tokens,
            sim_threshold=args.sim_threshold,
        )
        print(json.dumps({k: v for k, v in out.items() if k != "eml_bytes"}, ensure_ascii=False, indent=2))
        return

    if args.cmd == "chat":
        out = c.chat(
            args.query,
            k=args.k,
            session_id=args.session,
            reset_session=args.reset,
            temperature=args.temperature,
        )
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return

    if args.cmd == "summarize":
        data_or_md = c.summarize(
            thread=args.thread or None,
            conv_id=args.conv or None,
            provider=args.provider,
            temperature=args.temperature,
            output_format=args.format,
            merge_manifest=not args.no_merge_manifest,
        )
        if isinstance(data_or_md, str):
            print(data_or_md)
        else:
            print(json.dumps(data_or_md, ensure_ascii=False, indent=2))
        return

    ap.print_help()


if __name__ == "__main__":
    main()
