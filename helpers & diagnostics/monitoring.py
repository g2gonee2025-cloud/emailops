
#!/usr/bin/env python3
"""
Consolidated monitoring and statistics utilities for EmailOps.
Combines functionality from monitor.py, statistics.py, check_chunks.py, and live_test.py.
"""

from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import asyncio
import json
import logging
import os
import sys
import time

from datetime import UTC, datetime, timedelta
from emailops.core_config import get_config
from emailops.indexing_metadata import load_index_metadata
from emailops.util_main import find_conversation_dirs
import argparse
import psutil

# Try to import optional dependencies
try:
    import psutil
except ImportError:
    psutil = None

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
try:
    config = get_config()
    INDEX_DIRNAME = getattr(config, "INDEX_DIRNAME", "_index")
    CHUNK_DIRNAME = getattr(config, "CHUNK_DIRNAME", "_chunks")
    DEFAULT_CHUNK_SIZE = getattr(config, "DEFAULT_CHUNK_SIZE", 1600)
    DEFAULT_CHUNK_OVERLAP = getattr(config, "DEFAULT_CHUNK_OVERLAP", 200)
except ImportError:
    # Fallback if config module not available
    INDEX_DIRNAME = "_index"
    CHUNK_DIRNAME = "_chunks"
    DEFAULT_CHUNK_SIZE = 1600
    DEFAULT_CHUNK_OVERLAP = 200

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


def colored(text: str, color: str) -> str:
    """Apply color to text if supported"""
    if os.getenv("NO_COLOR"):
        return text
    try:
        return f"{color}{text}{Colors.ENDC}" if sys.stdout.isatty() else text
    except Exception:
        return text


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
    last_updated: str | None = None
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
# Index Monitor
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
        """Check current index status"""
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
        status.conversations_indexed = self._estimate_conversations_indexed(mapping_docs)

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
                try:
                    status.index_file_size_mb = round(p.stat().st_size / (1024 * 1024), 1)
                except:
                    pass
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
        """Analyze indexing rate and estimate completion time"""
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
        """Find Python processes that appear to be indexing"""
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
                    if any(keyword in cmd_str for keyword in ["index", "vertex", "embed", "chunk"]):
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

    # Private helper methods
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
        try:
            return len(find_conversation_dirs(self.root_dir))
        except:
            # Fallback: count directories with Conversation.txt
            count = 0
            try:
                for p in self.root_dir.rglob("Conversation.txt"):
                    if p.is_file():
                        count += 1
            except Exception:
                pass
            return count

    def _estimate_conversations_indexed(self, mapping_docs: list[dict[str, Any]]) -> int:
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
        try:
            meta = load_index_metadata(self.index_dir)
            if isinstance(meta, dict):
                status.provider = meta.get("provider")
                status.model = meta.get("model")
                status.actual_dimensions = meta.get("actual_dimensions") or meta.get("dimensions")
                status.index_type = meta.get("index_type")
        except:
            pass

    def _find_newest_artifact(self) -> tuple:
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
            sz = f"{status.index_file_size_mb:.1f} MB" if status.index_file_size_mb else "n/a"
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
            logger.warning(colored("\n⚠ No recent activity", Colors.YELLOW))


# -------------------------
# Chunk Analysis
# -------------------------

class ChunkAnalyzer:
    """Analyze chunk files and processing status."""

    def __init__(self, outlook_dir: Path | None = None):
        self.outlook_dir = outlook_dir or Path(os.getenv("OUTLOOK_EXPORT_ROOT", "C:/Users/ASUS/Desktop/Outlook"))
        self.chunks_dir = self.outlook_dir / CHUNK_DIRNAME / "chunks"
        self.log_dir = self.outlook_dir / CHUNK_DIRNAME / "_chunker_state"

    def analyze_chunks(self) -> dict[str, Any]:
        """Analyze chunk files and return statistics."""
        print_section("CHUNK DATA ANALYSIS")

        result = {
            "chunks_dir": str(self.chunks_dir),
            "exists": False,
            "total_files": 0,
            "total_chunks": 0,
            "empty_files": 0,
            "avg_chunks_per_file": 0,
            "chunk_sizes": {},
            "errors": []
        }

        if not self.chunks_dir.exists():
            print(f"❌ Chunks directory not found: {self.chunks_dir}")
            return result

        print(f"✅ Chunks directory found: {self.chunks_dir}")
        result["exists"] = True

        # Count chunk files
        chunk_files = list(self.chunks_dir.glob("*.json"))
        result["total_files"] = len(chunk_files)
        print(f"\nTotal chunk files: {len(chunk_files)}")

        if not chunk_files:
            return result

        # Analyze chunks
        total_chunks = 0
        empty_count = 0
        sizes = []

        # Sample first 100 files for statistics
        sample_files = chunk_files[:100]

        for chunk_file in sample_files:
            try:
                with Path.open(chunk_file, encoding='utf-8') as f:
                    data = json.load(f)

                if not data or data == []:
                    empty_count += 1
                    continue

                # Handle different formats
                if isinstance(data, dict):
                    chunks = data.get('chunks', [])
                elif isinstance(data, list):
                    chunks = data
                else:
                    continue

                total_chunks += len(chunks)

                # Get chunk sizes
                for chunk in chunks:
                    if isinstance(chunk, dict):
                        text = chunk.get('text', '')
                        sizes.append(len(text))

            except Exception as e:
                result["errors"].append(f"Error reading {chunk_file.name}: {e!s}")

        # Calculate statistics
        if sample_files:
            scale_factor = len(chunk_files) / len(sample_files)
            result["total_chunks"] = int(total_chunks * scale_factor)
            result["empty_files"] = int(empty_count * scale_factor)

            if result["total_files"] > 0:
                result["avg_chunks_per_file"] = result["total_chunks"] / result["total_files"]

        if sizes:
            result["chunk_sizes"] = {
                "min": min(sizes),
                "max": max(sizes),
                "avg": sum(sizes) / len(sizes)
            }

            print("\nChunk size statistics:")
            print(f"  - Min size: {result['chunk_sizes']['min']}")
            print(f"  - Max size: {result['chunk_sizes']['max']}")
            print(f"  - Avg size: {result['chunk_sizes']['avg']:.0f}")

        print(f"\nEmpty files: {result['empty_files']}")
        print(f"Total chunks (estimated): {result['total_chunks']}")
        print(f"Average chunks per file: {result['avg_chunks_per_file']:.1f}")

        return result

    def analyze_logs(self) -> dict[str, Any]:
        """Analyze chunking log files."""
        print_section("LOG FILE ANALYSIS")

        result = {
            "log_dir": str(self.log_dir),
            "exists": False,
            "total_logs": 0,
            "latest_log": None,
            "errors": []
        }

        if not self.log_dir.exists():
            print(f"❌ Log directory not found: {self.log_dir}")
            return result

        print(f"✅ Log directory found: {self.log_dir}")
        result["exists"] = True

        log_files = list(self.log_dir.glob("*.log"))
        result["total_logs"] = len(log_files)
        print(f"\nTotal log files: {len(log_files)}")

        if not log_files:
            return result

        # Get most recent log
        latest_log = max(log_files, key=lambda p: p.stat().st_mtime)
        result["latest_log"] = {
            "name": latest_log.name,
            "size_kb": latest_log.stat().st_size / 1024
        }

        print(f"\nMost recent log: {latest_log.name}")
        print(f"Size: {result['latest_log']['size_kb']:.1f} KB")

        # Read last few lines
        try:
            with Path.open(latest_log, encoding='utf-8') as f:
                lines = f.readlines()

            print("\nLast 10 log entries:")
            for line in lines[-10:]:
                print(f"  {line.strip()}")

            # Check for errors
            error_lines = [_l for _l in lines if 'ERROR' in _l or 'error' in _l.lower()]
            if error_lines:
                print(f"\n⚠️ Found {len(error_lines)} error entries in log")
                result["errors"] = error_lines[:10]  # Store first 10 errors
            else:
                print("\n✅ No errors found in log file")

        except Exception as e:
            print(f"❌ Error reading log file: {e}")
            result["errors"].append(str(e))

        return result


# -------------------------
# File Statistics
# -------------------------

class FileStatisticsAnalyzer:
    """Analyze file statistics and processing rules."""

    @staticmethod
    def analyze_file_processing() -> None:
        """Display detailed analysis of which files get chunked vs ignored."""
        print_section("EMAILOPS FILE PROCESSING ANALYSIS")

        logger.info("\n📊 YOUR OUTLOOK EXPORT STATISTICS")
        logger.info("-" * 80)
        logger.info("Total Files: 25,024")
        logger.info("Conversations: 3,369")
        logger.info("")

        logger.info("✅ FILES THAT GET CHUNKED (Processed & Indexed)")
        logger.info("-" * 80)

        chunked = {
            ".txt": (12923, "Text files (Conversation.txt + others)"),
            ".pdf": (4167, "PDF documents"),
            ".docx": (229, "Word documents (modern)"),
            ".doc": (45, "Word documents (legacy)"),
            ".xlsx": (736, "Excel spreadsheets (modern)"),
            ".xls": (76, "Excel spreadsheets (legacy)"),
            ".csv": (18, "CSV data files"),
        }

        total_chunked = 0
        for ext, (count, desc) in chunked.items():
            if count > 0:
                logger.info(f"  {ext:12} {count:5,} files - {desc}")
                total_chunked += count

        logger.info(f"\n  TOTAL:      {total_chunked:5,} files ({(total_chunked / 25024) * 100:.1f}% of all files)")

        logger.info("\n❌ FILES THAT GET IGNORED (Not Processed)")
        logger.info("-" * 80)

        ignored = {
            ".json": (3370, "Metadata files (manifest.json, summary.json)"),
            ".log": (3369, "Log files (system generated)"),
            ".zip": (52, "Compressed archives (not extracted)"),
            ".pptx": (19, "PowerPoint presentations (not supported)"),
            ".eml": (11, "Raw email files (not supported)"),
            ".msg": (4, "Outlook message files (not supported)"),
            ".rpmsg": (5, "Encrypted/protected messages (not supported)"),
        }

        total_ignored = 0
        for ext, (count, desc) in ignored.items():
            if count > 0:
                logger.info(f"  {ext:12} {count:5,} files - {desc}")
                total_ignored += count

        logger.info(f"\n  TOTAL:      {total_ignored:5,} files ({(total_ignored / 25024) * 100:.1f}% of all files)")

        logger.info("\n📈 SUMMARY:")
        logger.info(f"PROCESSED: {total_chunked:,} files ({(total_chunked / 25024) * 100:.1f}%)")
        logger.info(f"IGNORED: {total_ignored:,} files ({(total_ignored / 25024) * 100:.1f}%)")

    @staticmethod
    def get_file_statistics(root: Path | None = None) -> dict[str, Any]:
        """Generate comprehensive file statistics for Outlook export directory."""
        if root is None:
            root = Path(os.getenv("OUTLOOK_EXPORT_ROOT", "C:/Users/ASUS/Desktop/Outlook"))
        elif isinstance(root, str):
            root = Path(root)

        print_section("FILE STATISTICS ANALYZER")

        # Count conversation folders
        convos = [d for d in root.iterdir() if d.is_dir() and not d.name.startswith("_")]
        logger.info(f"\n📁 Conversation Folders: {len(convos):,}")

        # Count Conversation.txt files
        conv_txt = sum(1 for d in convos if (d / "Conversation.txt").exists())
        logger.info(f"📄 Conversation.txt Files: {conv_txt:,}")

        # Count all files by extension
        logger.info("\n📊 Scanning all files...")
        exts = Counter()
        total_files = 0

        for d in convos:
            for f in d.rglob("*"):
                if f.is_file():
                    total_files += 1
                    exts[f.suffix.lower() if f.suffix else "(no extension)"] += 1

        logger.info(f"✅ Total Files Scanned: {total_files:,}")

        logger.info("\n📈 Top 20 File Extensions:")
        logger.info("-" * 80)
        for ext, count in exts.most_common(20):
            pct = (count / total_files) * 100 if total_files > 0 else 0
            logger.info(f"  {ext:20} {count:8,} files ({pct:5.1f}%)")

        return {
            "root_path": str(root),
            "conversation_folders": len(convos),
            "conversation_txt_files": conv_txt,
            "total_files": total_files,
            "extensions": dict(exts),
            "top_extensions": dict(exts.most_common(20)),
        }


# -------------------------
# Live Testing
# -------------------------

class LiveTester:
    """Run live tests on conversation directories."""

    def __init__(self, outlook_dir: Path | None = None, limit: int = 100):
        self.outlook_dir = outlook_dir or Path("C:/Users/ASUS/Desktop/Outlook")
        self.limit = limit
        self.log_file = Path("log") / f"live_test_{int(time.time())}.log"
        self.log_file.parent.mkdir(exist_ok=True)

    async def run_test(self) -> dict[str, Any]:
        """Run live test on conversation directories."""
        logger.info("--- Starting Live Conversation Analysis Test ---")
        logger.info(f"Outlook Directory: {self.outlook_dir}")
        logger.info(f"Conversation Limit: {self.limit}")
        logger.info(f"Log file: {self.log_file}")

        result = {
            "outlook_dir": str(self.outlook_dir),
            "limit": self.limit,
            "success_count": 0,
            "error_count": 0,
            "total_processed": 0,
            "errors": []
        }

        if not self.outlook_dir.exists() or not self.outlook_dir.is_dir():
            logger.error(f"Outlook directory not found: {self.outlook_dir}")
            result["errors"].append("Outlook directory not found")
            return result

        try:
            from emailops.feature_summarize import analyze_conversation_dir, _atomic_write_text
            from emailops.util_main import find_conversation_dirs
        except ImportError as e:
            logger.error(f"Failed to import EmailOps modules: {e}")
            result["errors"].append(f"Import error: {e}")
            return result

        try:
            conversation_dirs = find_conversation_dirs(self.outlook_dir)
        except Exception as e:
            logger.error(f"Failed to find conversation directories: {e}")
            result["errors"].append(f"Failed to find conversations: {e}")
            return result

        if not conversation_dirs:
            logger.warning("No conversation directories found to process.")
            return result

        logger.info(f"Found {len(conversation_dirs)} total conversations. Processing up to {self.limit}.")

        for i, convo_dir in enumerate(conversation_dirs[:self.limit]):
            result["total_processed"] += 1
            logger.info(f"--- Processing Conversation {i+1}/{self.limit}: {convo_dir.name} ---")

            try:
                # Check if Conversation.txt exists
                if not (convo_dir / "Conversation.txt").exists():
                    logger.error(f"SKIPPING: Conversation.txt not found in {convo_dir.name}")
                    result["error_count"] += 1
                    continue

                # Analyze the conversation
                analysis_result = await analyze_conversation_dir(thread_dir=convo_dir)

                # Check for errors in the result
                if not analysis_result or "_metadata" not in analysis_result:
                    raise ValueError("Analysis result is empty or missing metadata.") from None

                # Log success and save result
                logger.info(f"SUCCESS: Analysis complete for {convo_dir.name}")

                # Save the analysis
                output_path = convo_dir / "summary_live_test.json"
                _atomic_write_text(output_path, json.dumps(analysis_result, indent=2, ensure_ascii=False))
                logger.info(f"Saved analysis to {output_path}")

                result["success_count"] += 1

            except Exception as e:
                logger.error(f"ERROR processing {convo_dir.name}: {e}", exc_info=True)
                result["error_count"] += 1
                result["errors"].append(f"{convo_dir.name}: {e!s}")

            logger.info(f"--- Finished Processing {convo_dir.name} ---")
            await asyncio.sleep(1)  # Small delay to avoid overwhelming APIs

        logger.info("--- Live Test Summary ---")
        logger.info(f"Total Conversations Processed: {result['total_processed']}")
        logger.info(f"Successful Analyses: {result['success_count']}")
        logger.info(f"Failed Analyses: {result['error_count']}")
        logger.info("--- Test Complete ---")

        return result


# -------------------------
# Main CLI
# -------------------------

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(description="Monitor and analyze EmailOps processing")

    parser.add_argument(
        "--root", default=str(Path.cwd()), help="Root directory containing index"
    )
    parser.add_argument(
        "--json", action="store_true", help="Output JSON instead of text"
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Commands
    subparsers.add_parser("status", help="Check index status")
    subparsers.add_parser("rate", help="Analyze indexing rate")
    subparsers.add_parser("procs", help="Find indexing processes")
    subparsers.add_parser("chunks", help="Analyze chunk files")
    subparsers.add_parser("files", help="Analyze file statistics")
    subparsers.add_parser("processing", help="Show file processing rules")
    subparsers.add_parser("live", help="Run live test on conversations")
    subparsers.add_parser("full", help="Full status report")

    # Process ID command
    pid_parser = subparsers.add_parser("pid", help="Check specific process")
    pid_parser.add_argument("pid", type=int, help="Process ID")

    args = parser.parse_args()

    # Default to status if no command
    if not args.command:
        args.command = "status"

    # Execute command
    if args.command == "status":
        monitor = IndexMonitor(root_dir=args.root)
        status = monitor.check_status(emit_text=not args.json)
        if args.json:
            print(json.dumps(status.to_dict(), indent=2))

    elif args.command == "rate":
        monitor = IndexMonitor(root_dir=args.root)
        analysis = monitor.analyze_rate(emit_text=not args.json)
        if args.json:
            print(json.dumps(analysis, indent=2))

    elif args.command == "procs":
        monitor = IndexMonitor(root_dir=args.root)
        processes = monitor.find_processes(emit_text=not args.json)
        if args.json:
            print(json.dumps([p.to_dict() for p in processes], indent=2))

    elif args.command == "pid":
        if psutil:
            try:
                proc = psutil.Process(args.pid)
                mem = proc.memory_info().rss / 1024 / 1024

                process = ProcessInfo(
                    pid=args.pid,
                    name=proc.name(),
                    command=" ".join(proc.cmdline())[:200],
                    memory_mb=round(mem, 1),
                    status=proc.status(),
                    working_dir=proc.cwd() if hasattr(proc, "cwd") else "unknown",
                )

                if args.json:
                    print(json.dumps(process.to_dict(), indent=2))
                else:
                    print(f"\nPID {args.pid}:")
                    print(f"  Name:     {process.name}")
                    print(f"  Command:  {process.command}")
                    print(f"  Memory:   {process.memory_mb:.1f} MB")
                    print(f"  Status:   {process.status}")
                    print(f"  CWD:      {process.working_dir}")
            except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                if not args.json:
                    print(f"Process {args.pid} not found or access denied: {e}")
        else:
            print("psutil not installed")

    elif args.command == "chunks":
        analyzer = ChunkAnalyzer()
        result = analyzer.analyze_chunks()
        log_result = analyzer.analyze_logs()
        if args.json:
            print(json.dumps({"chunks": result, "logs": log_result}, indent=2))

    elif args.command == "files":
        result = FileStatisticsAnalyzer.get_file_statistics()
        if args.json:
            print(json.dumps(result, indent=2))

    elif args.command == "processing":
        FileStatisticsAnalyzer.analyze_file_processing()

    elif args.command == "live":
        tester = LiveTester()
        result = asyncio.run(tester.run_test())
        if args.json:
            print(json.dumps(result, indent=2))

    elif args.command == "full":
        if args.json:
            monitor = IndexMonitor(root_dir=args.root)
            status = monitor.check_status(emit_text=False)
            analysis = monitor.analyze_rate(emit_text=False)
            processes = monitor.find_processes(emit_text=False)

            analyzer = ChunkAnalyzer()
            chunks = analyzer.analyze_chunks()

            print(json.dumps({
                "status": status.to_dict(),
                "rate": analysis,
                "processes": [p.to_dict() for p in processes],
                "chunks": chunks
            }, indent=2))
        else:
            monitor = IndexMonitor(root_dir=args.root)
            monitor.check_status(emit_text=True)
            monitor.analyze_rate(emit_text=True)
            monitor.find_processes(emit_text=True)

            analyzer = ChunkAnalyzer()
            analyzer.analyze_chunks()
            analyzer.analyze_logs()

    return 0


if __name__ == "__main__":
    sys.exit(main())
