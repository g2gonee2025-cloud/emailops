#!/usr/bin/env python3
"""
Vertex AI Utilities - Monitoring and Status Tools (production-ready)

This module provides a small, dependency-light toolbox to:
- Inspect the on-disk index produced by the indexer
- Estimate indexing throughput and ETA
- Inspect running "indexing" Python processes

Key behaviors:
- Robust, non-interactive CLI (subcommands) + optional interactive fallback
- Machine-readable JSON mode that never prints human text alongside JSON
- Safer file I/O with UTF‑8 BOM tolerance and defensive JSON parsing
- Accurate conversation totals using repository utilities when available
- Detects "active indexing" by also watching per-worker pickle files
- Works without psutil installed (process features degrade gracefully)
- Time handling is UTC-aware; human text uses safe local conversions

Compatible with indexes created by:
- email_indexer.py (full metadata)  (mapping.json + meta.json + embeddings.npy / index.faiss)
- vertex_indexer.py (parallel mode) (worker pickles under _index/embeddings)

Environment:
- INDEX_DIRNAME: name of the index directory (default: "_index")
- ACTIVE_WINDOW_SECONDS: seconds to consider "active" since last artifact write (default: 120)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# Optional psutil import (degrade gracefully if missing)
try:  # pragma: no cover
    import psutil  # type: ignore
except Exception:  # pragma: no cover
    psutil = None  # type: ignore

# Lazy imports from the repo (avoid heavy imports at module import time)
# These are available in this project; if not, we still degrade cleanly.
try:  # pragma: no cover
    from emailops.index_metadata import load_index_metadata  # type: ignore
except Exception:
    load_index_metadata = None  # type: ignore

try:  # pragma: no cover
    from emailops.utils import find_conversation_dirs  # type: ignore
except Exception:
    find_conversation_dirs = None  # type: ignore

# -------------------------
# Constants / Defaults
# -------------------------

INDEX_DIRNAME_DEFAULT = os.getenv("INDEX_DIRNAME", "_index")
ACTIVE_WINDOW_SECONDS_DEFAULT = int(os.getenv("ACTIVE_WINDOW_SECONDS", "120"))

# -------------------------
# TTY colors (no hard dependency on colorama)
# -------------------------
class Colors:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"


def _supports_color() -> bool:
    """Basic detection for ANSI color support."""
    if os.getenv("NO_COLOR"):
        return False
    try:
        return sys.stdout.isatty()
    except Exception:
        return False


def _c(text: str, color: str) -> str:
    return f"{color}{text}{Colors.ENDC}" if _supports_color() else text


def print_section(title: str) -> None:
    print(f"\n{'='*70}\n  {title}\n{'='*70}")


# -------------------------
# Data models
# -------------------------

@dataclass
class IndexStatus:
    root_dir: str
    index_dir: str
    index_exists: bool
    documents_indexed: int = 0
    conversations_total: int = 0
    conversations_indexed: int = 0
    progress_percent: float = 0.0
    last_updated: Optional[str] = None         # ISO-8601 (UTC)
    is_active: bool = False
    active_source: Optional[str] = None        # newest artifact filename
    index_file: Optional[str] = None           # primary artifact filename (faiss/embeddings/mapping)
    index_file_size_mb: Optional[float] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    actual_dimensions: Optional[int] = None
    index_type: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# -------------------------
# Core monitor
# -------------------------

class IndexingMonitor:
    """
    Monitor indexing progress and status.

    Parameters
    ----------
    root_dir : str | Path | None
        Export root containing conversation folders and the index directory.
    index_dirname : str
        Name of the index directory under root (default: "_index").
    active_window_seconds : int
        If the newest index-related file was modified more recently than this
        many seconds, we consider indexing "active".
    """

    def __init__(
        self,
        root_dir: Optional[os.PathLike[str] | str] = None,
        *,
        index_dirname: str = INDEX_DIRNAME_DEFAULT,
        active_window_seconds: int = ACTIVE_WINDOW_SECONDS_DEFAULT,
    ) -> None:
        self.root_dir = Path(root_dir or os.getcwd()).expanduser().resolve()
        self.index_dir = self.root_dir / index_dirname
        self.active_window = timedelta(seconds=max(10, active_window_seconds))

    # ---------- public API ---------- #

    def check_status(self, *, emit_text: bool = True) -> IndexStatus:
        """
        Compute a snapshot of current index status.

        Parameters
        ----------
        emit_text : bool
            If True, prints a human-readable summary. If False, returns only data (for JSON).
        """
        status = IndexStatus(
            root_dir=str(self.root_dir),
            index_dir=str(self.index_dir),
            index_exists=self.index_dir.exists(),
        )

        if not status.index_exists:
            if emit_text:
                print(_c("❌ No index directory found - indexing may not have started", Colors.RED))
            return status

        # 1) Load mapping.json to count documents
        mapping_docs = self._load_mapping()
        status.documents_indexed = len(mapping_docs)

        # 2) Count total conversations robustly using repo utils when available
        total = self._count_conversations()
        status.conversations_total = total

        # 3) Estimate conversations_indexed
        status.conversations_indexed = self._estimate_conversations_indexed(mapping_docs)

        # 4) Load meta for provider/model/dimensions/index_type
        self._populate_meta_fields(status)

        # 5) Determine the most recent index-related artifact
        newest_path, newest_mtime = self._find_newest_index_artifact()
        if newest_path and newest_mtime:
            # Store as UTC ISO-8601
            dt_utc = datetime.fromtimestamp(newest_mtime, tz=timezone.utc)
            status.last_updated = dt_utc.isoformat()
            status.active_source = newest_path.name

        # 6) Determine stable "primary" index artifact for display
        primary = None
        for candidate in ("index.faiss", "embeddings.npy", "mapping.json"):
            p = (self.index_dir / candidate)
            if p.exists():
                primary = p
                break
        if primary:
            status.index_file = primary.name
            try:
                status.index_file_size_mb = round(primary.stat().st_size / (1024 * 1024), 1)
            except Exception:
                status.index_file_size_mb = None

        # 7) Compute progress
        if status.conversations_total > 0:
            status.progress_percent = min(
                100.0,
                100.0 * status.conversations_indexed / status.conversations_total
            )

        # 8) Determine "active" based on recency of index artifacts
        if newest_mtime:
            # Compare in UTC to avoid naive/local mismatches
            now_utc = datetime.now(timezone.utc)
            dt_utc = datetime.fromtimestamp(newest_mtime, tz=timezone.utc)
            is_recent = (now_utc - dt_utc) <= self.active_window
            status.is_active = bool(is_recent)

        if emit_text:
            self._print_status(status)
        return status

    def analyze_rate(self, *, emit_text: bool = True) -> Dict[str, Any]:
        """
        Analyze indexing rate and ETA using meta.created_at when available.
        Returns a dict with rate_per_hour, eta_hours, and remaining_conversations.
        """
        status = self.check_status(emit_text=emit_text)
        if status.documents_indexed == 0 and status.conversations_indexed == 0:
            if emit_text:
                print(_c("No items processed yet", Colors.YELLOW))
            return {}

        created = self._infer_created_at()
        if not created:
            if emit_text:
                print(_c("Cannot determine start time (meta.json missing 'created_at').", Colors.YELLOW))
            return {}

        elapsed = (datetime.now(timezone.utc) - created).total_seconds()
        if elapsed <= 0:
            return {}

        # Prefer conversation-level rates if we have totals
        processed = status.conversations_indexed or status.documents_indexed
        rate_per_second = processed / elapsed
        rate_per_hour = rate_per_second * 3600.0

        remaining = max(0, (status.conversations_total or 0) - (status.conversations_indexed or 0))
        eta_seconds = remaining / rate_per_second if rate_per_second > 0 else 0.0
        eta_hours = eta_seconds / 3600.0

        if emit_text:
            print_section("INDEXING RATE ANALYSIS")
            print(f"Processed: {processed:,}")
            print(f"Elapsed:   {elapsed/3600.0:.2f} hours")
            print(f"Rate:      {rate_per_hour:.1f} items/hour")
            if status.conversations_total:
                print(f"Remaining: {remaining:,} conversations")
                print(f"ETA:       {eta_hours:.1f} hours")

        return {
            "rate_per_hour": rate_per_hour,
            "eta_hours": eta_hours,
            "remaining_conversations": remaining,
            "processed": processed,
            "elapsed_seconds": elapsed,
            "start_time": created.isoformat(),
        }

    # ---------- processes ---------- #

    @staticmethod
    def find_indexing_processes(*, emit_text: bool = True) -> List[Dict[str, Any]]:
        """
        Find Python processes that look like indexers (best effort).
        Returns a list of {pid, command, memory_mb}.
        """
        if psutil is None:
            # Do not print here to keep JSON mode clean; caller decides how to inform the user.
            return []

        results: List[Dict[str, Any]] = []
        for proc in psutil.process_iter(["pid", "name", "cmdline", "memory_info"]):
            try:
                name = (proc.info.get("name") or "").lower()
                cmdline = proc.info.get("cmdline") or []
                if ("python" in name) and any(
                    ("index" in str(arg).lower()) or
                    ("vertex" in str(arg).lower())
                    for arg in cmdline
                ):
                    mem = proc.info.get("memory_info")
                    mem_mb = (mem.rss / 1024 / 1024) if mem else 0.0
                    results.append({
                        "pid": proc.info["pid"],
                        "command": " ".join(cmdline),
                        "memory_mb": round(mem_mb, 1),
                    })
            except (psutil.NoSuchProcess, psutil.AccessDenied):  # type: ignore
                continue
        return results

    @staticmethod
    def check_process(pid: int) -> Dict[str, Any]:
        """Return details for a specific PID."""
        if psutil is None:
            return {"exists": False, "error": "psutil not installed"}
        try:
            p = psutil.Process(pid)  # type: ignore
            mem = p.memory_info().rss / 1024 / 1024
            return {
                "exists": True,
                "name": p.name(),
                "command": " ".join(p.cmdline()),
                "memory_mb": round(mem, 1),
                "status": p.status(),
                "working_dir": getattr(p, "cwd", lambda: "unknown")(),
            }
        except psutil.NoSuchProcess:  # type: ignore
            return {"exists": False}
        except Exception as e:
            return {"exists": False, "error": str(e)}

    # ---------- helpers ---------- #

    def _load_mapping(self) -> List[Dict[str, Any]]:
        mapping_path = self.index_dir / "mapping.json"
        if not mapping_path.exists():
            return []
        try:
            # tolerate BOM
            with mapping_path.open("r", encoding="utf-8-sig") as f:
                data = json.load(f)
            return data if isinstance(data, list) else []
        except Exception:
            return []

    def _count_conversations(self) -> int:
        """
        Count total conversations by finding Conversation.txt folders.
        Uses project utility when available; falls back to heuristic.
        """
        # Preferred: repository helper
        if callable(find_conversation_dirs):
            try:
                return len(find_conversation_dirs(self.root_dir))  # type: ignore[arg-type]
            except Exception:
                pass

        # Fallback heuristic: count directories with Conversation.txt
        count = 0
        try:
            for p in self.root_dir.rglob("Conversation.txt"):
                if p.is_file():
                    count += 1
        except Exception:
            pass
        return count

    @staticmethod
    def _estimate_conversations_indexed(mapping_docs: List[Dict[str, Any]]) -> int:
        """
        Estimate how many *conversations* (email threads) are represented in mapping.json.
        Prefer conv_id when present (new index), otherwise derive from id prefix.
        """
        conv_ids: set[str] = set()
        for d in mapping_docs:
            conv_id = d.get("conv_id")
            if conv_id:
                conv_ids.add(str(conv_id))
            else:
                doc_id = d.get("id")
                if isinstance(doc_id, str) and "::" in doc_id:
                    conv_ids.add(doc_id.split("::", 1)[0])
        return len(conv_ids)

    def _populate_meta_fields(self, status: IndexStatus) -> None:
        """Populate provider/model/dimensions/index_type from meta.json when available."""
        meta = None
        try:
            if callable(load_index_metadata):
                meta = load_index_metadata(self.index_dir)  # type: ignore[arg-type]
        except Exception:
            meta = None

        if isinstance(meta, dict):
            status.provider = (meta.get("provider") or None)
            status.model = (meta.get("model") or None)
            status.actual_dimensions = meta.get("actual_dimensions") or meta.get("dimensions") or None
            status.index_type = (meta.get("index_type") or None)

    def _find_newest_index_artifact(self) -> Tuple[Optional[Path], Optional[float]]:
        """
        Return (path, mtime) of the most recently-modified relevant index artifact among:
        - index.faiss
        - embeddings.npy
        - any worker pickle in _index/embeddings/
        - mapping.json
        - meta.json
        """
        candidates: List[Path] = []
        try:
            for name in ("index.faiss", "embeddings.npy", "mapping.json", "meta.json"):
                p = self.index_dir / name
                if p.exists():
                    candidates.append(p)
            # Worker outputs (parallel path)
            emb_dir = self.index_dir / "embeddings"
            if emb_dir.exists():
                for p in emb_dir.glob("worker_*_batch_*.pkl"):
                    candidates.append(p)
        except Exception:
            pass

        newest: Optional[Path] = None
        newest_mtime: Optional[float] = None

        for p in candidates:
            try:
                m = p.stat().st_mtime
            except Exception:
                continue
            if newest_mtime is None or m > newest_mtime:
                newest = p
                newest_mtime = m

        return newest, newest_mtime

    def _infer_created_at(self) -> Optional[datetime]:
        """Try to read meta.json created_at and return an aware UTC datetime."""
        meta_path = self.index_dir / "meta.json"
        if not meta_path.exists():
            return None
        try:
            with meta_path.open("r", encoding="utf-8") as f:
                meta = json.load(f)
            created_raw = meta.get("created_at")
            if not created_raw:
                return None
            # Handle trailing "Z" or plain ISO strings
            s = str(created_raw).replace("Z", "+00:00")
            created = datetime.fromisoformat(s)
            if created.tzinfo is None:
                created = created.replace(tzinfo=timezone.utc)
            return created.astimezone(timezone.utc)
        except Exception:
            return None

    @staticmethod
    def _print_status(status: IndexStatus) -> None:
        print_section("INDEXING STATUS")
        print(f"Root:       {status.root_dir}")
        print(f"Index dir:  {status.index_dir}")
        if not status.index_exists:
            return

        print(f"\nDocuments indexed:     {status.documents_indexed:,}")
        if status.conversations_total:
            pct = f"{status.progress_percent:.1f}%"
            print(f"Conversations indexed: {status.conversations_indexed:,} / {status.conversations_total:,}  ({pct})")
        else:
            print("Conversations indexed: (unknown total)")

        if status.index_file:
            sz = f"{status.index_file_size_mb:.1f} MB" if status.index_file_size_mb is not None else "n/a"
            print(f"Index artifact:        {status.index_file}  ({sz})")

        if status.active_source or status.last_updated:
            # Display last updated in local time for readability
            try:
                if status.last_updated:
                    dt = datetime.fromisoformat(status.last_updated)
                    local_dt = dt.astimezone() if dt.tzinfo else dt
                    ago = int((datetime.now(local_dt.tzinfo) - local_dt).total_seconds()) if dt.tzinfo else int((datetime.now() - dt).total_seconds())
                    print(f"Last updated:          {local_dt.strftime('%Y-%m-%d %H:%M:%S %Z')}  ({ago} seconds ago)")
                if status.active_source:
                    print(f"Newest artifact:       {status.active_source}")
            except Exception:
                pass

        if status.is_active:
            print(_c("\n✅ ACTIVELY INDEXING", Colors.GREEN))
        else:
            print(_c("\nℹ️  No recent writes detected", Colors.YELLOW))

        if status.provider or status.model or status.actual_dimensions or status.index_type:
            print("\nMetadata:")
            if status.provider:
                print(f"  Provider:   {status.provider}")
            if status.model:
                print(f"  Model:      {status.model}")
            if status.actual_dimensions:
                print(f"  Dimensions: {status.actual_dimensions}")
            if status.index_type:
                print(f"  Index type: {status.index_type}")


# -------------------------
# CLI
# -------------------------

def _interactive_menu(monitor: IndexingMonitor) -> int:
    """Interactive fallback (useful for quick local checks)."""
    print(_c("=" * 70, Colors.BOLD))
    print(_c("VERTEX AI UTILITIES", Colors.BOLD))
    print(_c("=" * 70, Colors.BOLD))
    print("\nOptions:")
    print("1. Check indexing status")
    print("2. Analyze indexing rate")
    print("3. Find indexing processes")
    print("4. Check specific process (PID)")
    print("5. Full status report")
    print("6. Exit")

    try:
        choice = input("\nSelect option (1-6): ").strip()
    except (EOFError, KeyboardInterrupt):
        return 0

    if choice == "1":
        monitor.check_status(emit_text=True)
    elif choice == "2":
        monitor.analyze_rate(emit_text=True)
    elif choice == "3":
        print_section("INDEXING PROCESSES")
        procs = monitor.find_indexing_processes(emit_text=True)
        if procs:
            print(f"Found {len(procs)} process(es):\n")
            for p in procs:
                print(f"PID {p['pid']}: {p['memory_mb']:.1f} MB\n  Command: {p['command'][:120]}...")
        else:
            print("No indexing processes found")
    elif choice == "4":
        try:
            pid = int(input("Enter process ID: ").strip())
        except Exception:
            print("Invalid PID")
            return 1
        info = monitor.check_process(pid)
        if info.get("exists"):
            print(f"\nPID {pid} found:")
            print(f"  Name:    {info.get('name')}")
            print(f"  Command: {str(info.get('command'))[:160]}...")
            print(f"  Memory:  {info.get('memory_mb'):.1f} MB")
            print(f"  Status:  {info.get('status')}")
            print(f"  CWD:     {info.get('working_dir')}")
        else:
            print(f"PID {pid} not found")
    elif choice == "5":
        # Full status + rate + processes
        status = monitor.check_status(emit_text=True)
        if (status.documents_indexed or status.conversations_indexed):
            monitor.analyze_rate(emit_text=True)
        print_section("RUNNING PROCESSES")
        procs = monitor.find_indexing_processes(emit_text=True)
        if procs:
            for p in procs:
                print(f"PID {p['pid']}: {p['memory_mb']:.1f} MB")
        else:
            print("No indexing processes found")
    elif choice == "6":
        print("Exiting...")
        return 0
    else:
        print("Invalid option")
        return 1
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(argv) if argv is not None else sys.argv[1:]
    ap = argparse.ArgumentParser(description="Vertex AI Utilities - monitor and status tools")
    ap.add_argument("--root", default=os.getcwd(), help="Export root containing conversations and the index directory")
    ap.add_argument("--index-dirname", default=INDEX_DIRNAME_DEFAULT, help="Index directory name (default: %(default)s)")
    ap.add_argument("--json", action="store_true", help="Emit machine-readable JSON instead of human text")

    # Subcommands (non-interactive)
    sub = ap.add_subparsers(dest="cmd")

    sub.add_parser("status", help="Print index status")
    sub.add_parser("rate", help="Analyze indexing throughput/ETA")
    sub.add_parser("procs", help="List indexing-related Python processes")
    p_pid = sub.add_parser("pid", help="Show details for a specific process id")
    p_pid.add_argument("pid", type=int, help="Process ID")

    sub.add_parser("full", help="Full status report (status + rate + processes)")

    args = ap.parse_args(argv)

    monitor = IndexingMonitor(
        args.root,
        index_dirname=args.index_dirname,
        active_window_seconds=ACTIVE_WINDOW_SECONDS_DEFAULT,
    )

    # Interactive fallback when no subcommand provided
    if args.cmd is None:
        if args.json:
            status = monitor.check_status(emit_text=False)
            print(json.dumps(status.to_dict(), ensure_ascii=False, indent=2))
            return 0
        return _interactive_menu(monitor)

    if args.cmd == "status":
        status = monitor.check_status(emit_text=not args.json)
        if args.json:
            print(json.dumps(status.to_dict(), ensure_ascii=False, indent=2))
        return 0

    if args.cmd == "rate":
        out = monitor.analyze_rate(emit_text=not args.json)
        if args.json:
            print(json.dumps(out, ensure_ascii=False, indent=2))
        return 0

    if args.cmd == "procs":
        procs = monitor.find_indexing_processes(emit_text=False)  # never print from helper
        if args.json:
            print(json.dumps(procs, ensure_ascii=False, indent=2))
        else:
            print_section("INDEXING PROCESSES")
            if procs:
                print(f"Found {len(procs)} process(es):\n")
                for p in procs:
                    print(f"PID {p['pid']}: {p['memory_mb']:.1f} MB\n  Command: {p['command'][:120]}...")
            else:
                # Only print human notice for non-JSON
                if psutil is None:
                    print(_c("psutil not installed; process inspection is unavailable.", Colors.YELLOW))
                else:
                    print("No indexing processes found")
        return 0

    if args.cmd == "pid":
        info = monitor.check_process(args.pid)
        if args.json:
            print(json.dumps(info, ensure_ascii=False, indent=2))
        else:
            if info.get("exists"):
                print(f"\nPID {args.pid} found:")
                print(f"  Name:    {info.get('name')}")
                print(f"  Command: {str(info.get('command'))[:160]}...")
                print(f"  Memory:  {info.get('memory_mb'):.1f} MB")
                print(f"  Status:  {info.get('status')}")
                print(f"  CWD:     {info.get('working_dir')}")
            else:
                if "error" in info:
                    print(_c(info.get("error", "Unknown error"), Colors.YELLOW))
                else:
                    print(f"PID {args.pid} not found")
        return 0

    if args.cmd == "full":
        status = monitor.check_status(emit_text=not args.json)
        if args.json:
            payload = {
                "status": status.to_dict(),
                "rate": monitor.analyze_rate(emit_text=False),
                "processes": monitor.find_indexing_processes(emit_text=False),
            }
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            return 0
        if status.documents_indexed or status.conversations_indexed:
            monitor.analyze_rate(emit_text=True)
        print_section("RUNNING PROCESSES")
        procs = monitor.find_indexing_processes(emit_text=False)
        if procs:
            for p in procs:
                print(f"PID {p['pid']}: {p['memory_mb']:.1f} MB")
        else:
            if psutil is None:
                print(_c("psutil not installed; process inspection is unavailable.", Colors.YELLOW))
            else:
                print("No indexing processes found")
        return 0

    ap.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
