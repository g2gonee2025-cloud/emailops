#!/usr/bin/env python3
"""
Monitoring and Status Tools for EmailOps Processing
Provides utilities to monitor indexing progress, analyze rates, and inspect processes
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

# Optional psutil import
try:
    import psutil
except ImportError:
    psutil = None

# Optional imports from emailops
try:
    from emailops.index_metadata import load_index_metadata
except ImportError:
    load_index_metadata = None

try:
    from diagnostics.utils import setup_logging
    from emailops.utils import find_conversation_dirs
except ImportError:
    find_conversation_dirs = None
    setup_logging = None

# Import centralized configuration
try:
    from emailops.config import get_config
    config = get_config()
    INDEX_DIRNAME = config.INDEX_DIRNAME
except ImportError:
    # Fallback if config module not available
    INDEX_DIRNAME = os.getenv("INDEX_DIRNAME", "_index")

logger = setup_logging() if setup_logging else logging.getLogger(__name__)


# -------------------------
# Constants
# -------------------------

# ACTIVE_WINDOW_SECONDS not in config, keep as environment variable
ACTIVE_WINDOW_SECONDS = int(os.getenv("ACTIVE_WINDOW_SECONDS", "120"))


# -------------------------
# Terminal Colors
# -------------------------


class Colors:
    """ANSI color codes for terminal output"""

    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"


def supports_color() -> bool:
    """Check if terminal supports ANSI colors"""
    if os.getenv("NO_COLOR"):
        return False
    try:
        return sys.stdout.isatty()
    except Exception:
        return False


def colored(text: str, color: str) -> str:
    """Apply color to text if supported"""
    return f"{color}{text}{Colors.ENDC}" if supports_color() else text


def print_section(title: str) -> None:
    """Print a formatted section header"""
    logger.info(f"\n{'=' * 70}\n  {title}\n{'=' * 70}")


# -------------------------
# Data Models
# -------------------------


@dataclass
class IndexStatus:
    """Status information for an index"""

    root_dir: str
    index_dir: str
    index_exists: bool
    documents_indexed: int = 0
    conversations_total: int = 0
    conversations_indexed: int = 0
    progress_percent: float = 0.0
    last_updated: str | None = None  # ISO-8601 (UTC)
    is_active: bool = False
    active_source: str | None = None
    index_file: str | None = None
    index_file_size_mb: float | None = None
    provider: str | None = None
    model: str | None = None
    actual_dimensions: int | None = None
    index_type: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ProcessInfo:
    """Information about a running process"""

    pid: int
    name: str
    command: str
    memory_mb: float
    status: str = "unknown"
    working_dir: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# -------------------------
# Main Monitor Class
# -------------------------


class IndexMonitor:
    """Monitor indexing progress and status"""

    def __init__(
        self,
        root_dir: str | None = None,
        index_dirname: str = INDEX_DIRNAME,
        active_window_seconds: int = ACTIVE_WINDOW_SECONDS,
    ):
        self.root_dir = Path(root_dir or Path.cwd()).expanduser().resolve()
        self.index_dir = self.root_dir / index_dirname
        self.active_window = timedelta(seconds=max(10, active_window_seconds))

    def check_status(self, emit_text: bool = True) -> IndexStatus:
        """
        Check current index status

        Args:
            emit_text: Whether to print human-readable output

        Returns:
            IndexStatus object with current status
        """
        status = IndexStatus(
            root_dir=str(self.root_dir),
            index_dir=str(self.index_dir),
            index_exists=self.index_dir.exists(),
        )

        if not status.index_exists:
            if emit_text:
                logger.error(colored("❌ No index directory found", Colors.RED))
            return status

        # Load mapping to count documents
        mapping_docs = self._load_mapping()
        status.documents_indexed = len(mapping_docs)

        # Count total conversations
        status.conversations_total = self._count_conversations()

        # Estimate conversations indexed
        status.conversations_indexed = self._estimate_conversations_indexed(
            mapping_docs
        )

        # Load metadata
        self._populate_meta_fields(status)

        # Find most recent artifact
        newest_path, newest_mtime = self._find_newest_artifact()
        if newest_path and newest_mtime:
            dt_utc = datetime.fromtimestamp(newest_mtime, tz=UTC)
            status.last_updated = dt_utc.isoformat()
            status.active_source = newest_path.name

        # Find primary index file
        for candidate in ("index.faiss", "embeddings.npy", "mapping.json"):
            p = self.index_dir / candidate
            if p.exists():
                status.index_file = candidate
                with contextlib.suppress(Exception):
                    status.index_file_size_mb = round(
                        p.stat().st_size / (1024 * 1024), 1
                    )
                break

        # Calculate progress
        if status.conversations_total > 0:
            status.progress_percent = min(
                100.0, 100.0 * status.conversations_indexed / status.conversations_total
            )

        # Check if actively indexing
        if newest_mtime:
            now_utc = datetime.now(UTC)
            dt_utc = datetime.fromtimestamp(newest_mtime, tz=UTC)
            status.is_active = (now_utc - dt_utc) <= self.active_window

        if emit_text:
            self._print_status(status)

        return status

    def analyze_rate(self, emit_text: bool = True) -> dict[str, Any]:
        """
        Analyze indexing rate and estimate completion time

        Args:
            emit_text: Whether to print human-readable output

        Returns:
            Dictionary with rate analysis
        """
        status = self.check_status(emit_text=False)

        if status.documents_indexed == 0:
            if emit_text:
                logger.warning(colored("No items processed yet", Colors.YELLOW))
            return {}

        # Try to get start time from metadata
        created = self._get_created_time()
        if not created:
            if emit_text:
                logger.warning(colored("Cannot determine start time", Colors.YELLOW))
            return {}

        elapsed = (datetime.now(UTC) - created).total_seconds()
        if elapsed <= 0:
            return {}

        # Calculate rates
        processed = status.conversations_indexed or status.documents_indexed
        rate_per_second = processed / elapsed
        rate_per_hour = rate_per_second * 3600.0

        remaining = max(0, status.conversations_total - status.conversations_indexed)
        eta_seconds = remaining / rate_per_second if rate_per_second > 0 else 0.0
        eta_hours = eta_seconds / 3600.0

        if emit_text:
            print_section("INDEXING RATE ANALYSIS")
            logger.info(f"Processed: {processed:,}")
            logger.info(f"Elapsed:   {elapsed / 3600.0:.2f} hours")
            logger.info(f"Rate:      {rate_per_hour:.1f} items/hour")
            if status.conversations_total:
                logger.info(f"Remaining: {remaining:,} conversations")
                logger.info(f"ETA:       {eta_hours:.1f} hours")

        return {
            "rate_per_hour": rate_per_hour,
            "eta_hours": eta_hours,
            "remaining": remaining,
            "processed": processed,
            "elapsed_seconds": elapsed,
            "start_time": created.isoformat(),
        }

    def find_processes(self, emit_text: bool = True) -> list[ProcessInfo]:
        """
        Find Python processes that appear to be indexing

        Args:
            emit_text: Whether to print human-readable output

        Returns:
            List of ProcessInfo objects
        """
        if psutil is None:
            if emit_text:
                logger.warning(colored("psutil not installed", Colors.YELLOW))
            return []

        processes = []

        for proc in psutil.process_iter(["pid", "name", "cmdline", "memory_info"]):
            try:
                name = (proc.info.get("name") or "").lower()
                cmdline = proc.info.get("cmdline") or []

                # Check if this looks like an indexing process
                if "python" in name:
                    cmd_str = " ".join(cmdline).lower()
                    if any(
                        keyword in cmd_str
                        for keyword in ["index", "vertex", "embed", "chunk"]
                    ):
                        mem = proc.info.get("memory_info")
                        mem_mb = (mem.rss / 1024 / 1024) if mem else 0.0

                        processes.append(
                            ProcessInfo(
                                pid=proc.info["pid"],
                                name=proc.info.get("name", "unknown"),
                                command=" ".join(cmdline)[:200],
                                memory_mb=round(mem_mb, 1),
                            )
                        )

            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        if emit_text and processes:
            print_section("INDEXING PROCESSES")
            for p in processes:
                logger.info(f"PID {p.pid}: {p.memory_mb:.1f} MB")
                logger.info(f"  Command: {p.command}")

        return processes

    def check_process(self, pid: int) -> ProcessInfo | None:
        """
        Get information about a specific process

        Args:
            pid: Process ID to check

        Returns:
            ProcessInfo if found, None otherwise
        """
        if psutil is None:
            return None

        try:
            proc = psutil.Process(pid)
            mem = proc.memory_info().rss / 1024 / 1024

            return ProcessInfo(
                pid=pid,
                name=proc.name(),
                command=" ".join(proc.cmdline())[:200],
                memory_mb=round(mem, 1),
                status=proc.status(),
                working_dir=proc.cwd() if hasattr(proc, "cwd") else "unknown",
            )
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return None

    # ---- Private Helper Methods ----

    def _load_mapping(self) -> list[dict[str, Any]]:
        """Load mapping.json if it exists"""
        mapping_path = self.index_dir / "mapping.json"
        if not mapping_path.exists():
            return []

        try:
            with mapping_path.open(encoding="utf-8-sig") as f:
                data = json.load(f)
            return data if isinstance(data, list) else []
        except Exception:
            return []

    def _count_conversations(self) -> int:
        """Count total conversations in root directory"""
        if callable(find_conversation_dirs):
            try:
                return len(find_conversation_dirs(self.root_dir))
            except Exception:
                pass

        # Fallback: count directories with Conversation.txt
        count = 0
        try:
            for p in self.root_dir.rglob("Conversation.txt"):
                if p.is_file():
                    count += 1
        except Exception:
            pass

        return count

    def _estimate_conversations_indexed(
        self, mapping_docs: list[dict[str, Any]]
    ) -> int:
        """Estimate number of conversations in mapping"""
        conv_ids = set()

        for doc in mapping_docs:
            conv_id = doc.get("conv_id")
            if conv_id:
                conv_ids.add(str(conv_id))
            else:
                doc_id = doc.get("id")
                if isinstance(doc_id, str) and "::" in doc_id:
                    conv_ids.add(doc_id.split("::", 1)[0])

        return len(conv_ids)

    def _populate_meta_fields(self, status: IndexStatus) -> None:
        """Load metadata from meta.json"""
        if callable(load_index_metadata):
            try:
                meta = load_index_metadata(self.index_dir)
                if isinstance(meta, dict):
                    status.provider = meta.get("provider")
                    status.model = meta.get("model")
                    status.actual_dimensions = meta.get(
                        "actual_dimensions"
                    ) or meta.get("dimensions")
                    status.index_type = meta.get("index_type")
            except Exception:
                pass

    def _find_newest_artifact(self) -> tuple[Path | None, float | None]:
        """Find the most recently modified index artifact"""
        candidates = []

        try:
            # Check main index files
            for name in ("index.faiss", "embeddings.npy", "mapping.json", "meta.json"):
                p = self.index_dir / name
                if p.exists():
                    candidates.append(p)

            # Check worker output files
            emb_dir = self.index_dir / "embeddings"
            if emb_dir.exists():
                for p in emb_dir.glob("worker_*_batch_*.pkl"):
                    candidates.append(p)
        except Exception:
            pass

        newest = None
        newest_mtime = None

        for p in candidates:
            try:
                mtime = p.stat().st_mtime
                if newest_mtime is None or mtime > newest_mtime:
                    newest = p
                    newest_mtime = mtime
            except Exception:
                continue

        return newest, newest_mtime

    def _get_created_time(self) -> datetime | None:
        """Get index creation time from metadata"""
        meta_path = self.index_dir / "meta.json"
        if not meta_path.exists():
            return None

        try:
            with meta_path.open(encoding="utf-8") as f:
                meta = json.load(f)

            created_raw = meta.get("created_at")
            if not created_raw:
                return None

            # Parse ISO format
            s = str(created_raw).replace("Z", "+00:00")
            created = datetime.fromisoformat(s)

            if created.tzinfo is None:
                created = created.replace(tzinfo=UTC)

            return created.astimezone(UTC)
        except Exception:
            return None

    def _print_status(self, status: IndexStatus) -> None:
        """Print formatted status information"""
        print_section("INDEXING STATUS")
        logger.info(f"Root:       {status.root_dir}")
        logger.info(f"Index dir:  {status.index_dir}")

        if not status.index_exists:
            return

        logger.info(f"\nDocuments indexed:     {status.documents_indexed:,}")

        if status.conversations_total:
            pct = f"{status.progress_percent:.1f}%"
            logger.info(
                f"Conversations indexed: {status.conversations_indexed:,} / {status.conversations_total:,}  ({pct})"
            )
        else:
            logger.warning("Conversations indexed: (unknown total)")

        if status.index_file:
            sz = (
                f"{status.index_file_size_mb:.1f} MB"
                if status.index_file_size_mb
                else "n/a"
            )
            logger.info(f"Index artifact:        {status.index_file}  ({sz})")

        if status.last_updated:
            try:
                dt = datetime.fromisoformat(status.last_updated)
                local_dt = dt.astimezone()
                ago = int((datetime.now(UTC) - dt).total_seconds())
                logger.info(
                    f"Last updated:          {local_dt.strftime('%Y-%m-%d %H:%M:%S %Z')}  ({ago} seconds ago)"
                )
            except Exception:
                pass

        if status.is_active:
            logger.info(colored("\n✅ ACTIVELY INDEXING", Colors.GREEN))
        else:
            logger.warning(colored("\ni  No recent activity", Colors.YELLOW))

        if any(
            [status.provider, status.model, status.actual_dimensions, status.index_type]
        ):
            logger.info("\nMetadata:")
            if status.provider:
                logger.info(f"  Provider:   {status.provider}")
            if status.model:
                logger.info(f"  Model:      {status.model}")
            if status.actual_dimensions:
                logger.info(f"  Dimensions: {status.actual_dimensions}")
            if status.index_type:
                logger.info(f"  Index type: {status.index_type}")


# -------------------------
# CLI Entry Point
# -------------------------


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Monitor indexing and processing operations"
    )

    parser.add_argument(
        "--root", default=str(Path.cwd()), help="Root directory containing index"
    )
    parser.add_argument(
        "--index-dir", default=INDEX_DIRNAME, help="Index directory name"
    )
    parser.add_argument(
        "--json", action="store_true", help="Output JSON instead of text"
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Status command
    subparsers.add_parser("status", help="Check index status")

    # Rate command
    subparsers.add_parser("rate", help="Analyze indexing rate")

    # Processes command
    subparsers.add_parser("procs", help="Find indexing processes")

    # Process ID command
    pid_parser = subparsers.add_parser("pid", help="Check specific process")
    pid_parser.add_argument("pid", type=int, help="Process ID")

    # Full report command
    subparsers.add_parser("full", help="Full status report")

    args = parser.parse_args()

    # Create monitor
    monitor = IndexMonitor(
        root_dir=args.root,
        index_dirname=args.index_dir,
    )

    # Default to status if no command
    if not args.command:
        args.command = "status"

    # Execute command
    if args.command == "status":
        status = monitor.check_status(emit_text=not args.json)
        if args.json:
            logger.info(json.dumps(status.to_dict(), indent=2))

    elif args.command == "rate":
        analysis = monitor.analyze_rate(emit_text=not args.json)
        if args.json:
            logger.info(json.dumps(analysis, indent=2))

    elif args.command == "procs":
        processes = monitor.find_processes(emit_text=not args.json)
        if args.json:
            logger.info(json.dumps([p.to_dict() for p in processes], indent=2))

    elif args.command == "pid":
        process = monitor.check_process(args.pid)
        if process:
            if args.json:
                logger.info(json.dumps(process.to_dict(), indent=2))
            else:
                logger.info(f"\nPID {args.pid}:")
                logger.info(f"  Name:     {process.name}")
                logger.info(f"  Command:  {process.command}")
                logger.info(f"  Memory:   {process.memory_mb:.1f} MB")
                logger.info(f"  Status:   {process.status}")
                logger.info(f"  CWD:      {process.working_dir}")
        else:
            if not args.json:
                logger.error(f"Process {args.pid} not found")

    elif args.command == "full":
        if args.json:
            status = monitor.check_status(emit_text=False)
            analysis = monitor.analyze_rate(emit_text=False)
            processes = monitor.find_processes(emit_text=False)

            logger.info(
                json.dumps(
                    {
                        "status": status.to_dict(),
                        "rate": analysis,
                        "processes": [p.to_dict() for p in processes],
                    },
                    indent=2,
                )
            )
        else:
            monitor.check_status(emit_text=True)
            monitor.analyze_rate(emit_text=True)
            monitor.find_processes(emit_text=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
