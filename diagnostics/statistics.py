#!/usr/bin/env python3
"""
Consolidated analysis and monitoring functionality for file statistics, chunk counting, and indexing progress.
Consolidates functionality from file_processing_analysis.py, file_stats.py, count_chunks.py, and monitor_indexing.py.
"""

import os
import pickle
import sys
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from diagnostics.utils import get_index_path, setup_logging

# Import centralized configuration
try:
    from emailops.config import get_config
    config = get_config()
    INDEX_DIRNAME = config.INDEX_DIRNAME
    CHUNK_DIRNAME = config.CHUNK_DIRNAME
    DEFAULT_CHUNK_SIZE = config.DEFAULT_CHUNK_SIZE
    DEFAULT_CHUNK_OVERLAP = config.DEFAULT_CHUNK_OVERLAP
except ImportError:
    # Fallback if config module not available
    INDEX_DIRNAME = "_index"
    CHUNK_DIRNAME = "_chunks"
    DEFAULT_CHUNK_SIZE = 1600
    DEFAULT_CHUNK_OVERLAP = 200

# Setup logging
logger = setup_logging()


def analyze_file_processing() -> None:
    """
    Display detailed analysis of which files get chunked vs ignored in EmailOps.
    Shows processing rules and statistics for different file types.
    """
    logger.info("=" * 80)
    logger.info("EMAILOPS FILE PROCESSING ANALYSIS")
    logger.info("=" * 80)

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
        ".md": (0, "Markdown files (if any)"),
        ".html/.htm": (0, "HTML files (if any)"),
        ".xml": (0, "XML files (if any)"),
        ".csv": (18, "CSV data files"),
    }

    total_chunked = 0
    for ext, (count, desc) in chunked.items():
        if count > 0:
            logger.info(f"  {ext:12} {count:5,} files - {desc}")
            total_chunked += count

    logger.info(
        f"\n  TOTAL:      {total_chunked:5,} files ({(total_chunked / 25024) * 100:.1f}% of all files)"
    )

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

    logger.info(
        f"\n  TOTAL:      {total_ignored:5,} files ({(total_ignored / 25024) * 100:.1f}% of all files)"
    )

    logger.info("\n" + "=" * 80)
    logger.info("DETAILED EXTRACTION RULES")
    logger.info("=" * 80)

    logger.info("\n1. CONVERSATION.TXT FILES")
    logger.info("-" * 80)
    logger.info("  Always chunked (primary content)")
    logger.info("  Each conversation -> 1 Conversation.txt")
    logger.info("  Your data: 3,369 files")
    logger.info("  Average chunks per conversation: ~38.3")

    logger.info("\n2. TEXT-BASED FORMATS (Extracted Fully)")
    logger.info("-" * 80)
    logger.info("  .txt, .md, .log, .json, .yaml, .yml, .csv")
    logger.info("  .html, .htm, .xml (tags stripped)")
    logger.info("  Encoding: UTF-8 with Latin-1 fallback")

    logger.info("\n3. MICROSOFT OFFICE FORMATS")
    logger.info("-" * 80)
    logger.info("  .docx (python-docx)")
    logger.info("     - Extracts paragraphs and tables")
    logger.info("     - Your data: 229 files")
    logger.info("")
    logger.info("  .doc (pywin32 on Windows, docx2txt fallback)")
    logger.info("     - Requires Microsoft Word on Windows for best results")
    logger.info("     - Your data: 45 files")
    logger.info("")
    logger.info("  .xlsx (pandas + openpyxl)")
    logger.info("     - Extracts all sheets as text")
    logger.info("     - Your data: 736 files")
    logger.info("")
    logger.info("  .xls (pandas + xlrd)")
    logger.info("     - Extracts all sheets as text")
    logger.info("     - Your data: 76 files")
    logger.info("")
    logger.info("  .pptx (NOT SUPPORTED)")
    logger.info("     - Ignored during chunking")
    logger.info("     - Your data: 19 files")

    logger.info("\n4. PDF DOCUMENTS")
    logger.info("-" * 80)
    logger.info("  .pdf (pypdf)")
    logger.info("     - Extracts text from all pages")
    logger.info("     - Skips encrypted PDFs")
    logger.info("     - Your data: 4,167 files (16.7% of all files!)")

    logger.info("\n5. SPECIAL FILES")
    logger.info("-" * 80)
    logger.info("  .json (manifest.json, summary.json)")
    logger.info("     - Used for metadata only, not chunked")
    logger.info("     - Your data: 3,370 files")
    logger.info("")
    logger.info("  .log files")
    logger.info("     - System-generated, ignored")
    logger.info("     - Your data: 3,369 files")
    logger.info("")
    logger.info("  .zip archives")
    logger.info("     - Not extracted or processed")
    logger.info("     - Your data: 52 files")
    logger.info("")
    logger.info("  .eml, .msg, .rpmsg")
    logger.info("     - Email formats not currently supported")
    logger.info("     - Your data: 20 files total")

    logger.info("\n" + "=" * 80)
    logger.info("CHUNKING PROCESS")
    logger.info("=" * 80)

    logger.info("""
1. Each Conversation.txt is ALWAYS processed
2. Attachments in Attachments/ folder are scanned
3. Supported file types have text extracted
4. Files > 8,000 characters are split into chunks
5. Each chunk is {DEFAULT_CHUNK_SIZE} characters with {DEFAULT_CHUNK_OVERLAP}-char overlap
6. Chunks are saved to JSON files in {CHUNK_DIRNAME}/chunks/

Your Processing Results:
  - 3,369 conversations processed
  - 12,924 chunk files created
  - 129,055 total chunks
  - Average 38.3 chunks per conversation
  - Average 10 chunks per JSON file
""")

    logger.info("=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    logger.info(f"""
PROCESSED: {total_chunked:,} files ({(total_chunked / 25024) * 100:.1f}%)
  - All Conversation.txt files
  - PDFs (4,167 files - your largest category!)
  - Office documents (1,086 files total)
  - Text/CSV files

IGNORED: {total_ignored:,} files ({(total_ignored / 25024) * 100:.1f}%)
  - Metadata JSON files
  - Log files
  - Unsupported formats (PowerPoint, archives, .eml)

KEY INSIGHT:
Your system is processing {(total_chunked / 25024) * 100:.1f}% of files, which is EXCELLENT!
The 4,167 PDF files being chunked is particularly valuable for search.
""")

    logger.info("=" * 80)


def get_file_statistics(root: Path | None = None) -> dict[str, Any]:
    """
    Generate comprehensive file statistics for Outlook export directory.

    Args:
        root: Path to Outlook export root. If not provided, uses default from environment

    Returns:
        Dictionary with file statistics by extension and counts
    """
    if root is None:
        root = Path(os.getenv("OUTLOOK_EXPORT_ROOT", "C:/Users/ASUS/Desktop/Outlook"))
    elif isinstance(root, str):
        root = Path(root)

    logger.info("=" * 80)
    logger.info("FILE STATISTICS ANALYZER")
    logger.info("=" * 80)

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

    logger.info("\n" + "=" * 80)

    # Return statistics dictionary
    return {
        "root_path": str(root),
        "conversation_folders": len(convos),
        "conversation_txt_files": conv_txt,
        "total_files": total_files,
        "extensions": dict(exts),
        "top_extensions": dict(exts.most_common(20)),
    }


def count_total_chunks(export_dir: str) -> int:
    """
    Count total chunks in embeddings directory by loading worker output files.

    Args:
        export_dir: Path to the main export directory

    Returns:
        Total chunk count, or -1 if error occurred
    """
    try:
        export_path = Path(export_dir)
        emb_dir = export_path / INDEX_DIRNAME / "embeddings"
        total_chunks = 0

        if not emb_dir.is_dir():
            logger.error(f"Error: Embeddings directory not found at {emb_dir}")
            return -1

        pickle_files = sorted(emb_dir.glob("worker_*_batch_*.pkl"))

        if not pickle_files:
            logger.warning("No processed batch files found yet.")
            return 0

        for pkl_file in pickle_files:
            try:
                with pkl_file.open("rb") as f:
                    data = pickle.load(f)
                    # The 'chunks' key holds a list of the actual chunk dictionaries
                    num_chunks_in_batch = len(data.get("chunks", []))
                    total_chunks += num_chunks_in_batch
            except Exception as e:
                logger.error(f"Could not process file {pkl_file}: {e}")
                continue

        logger.info(f"Total chunks created so far: {total_chunks:,}")
        return total_chunks

    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        return -1


def monitor_indexing_progress(log_file: Path | None = None) -> dict[str, Any]:
    """
    Monitor real-time indexing progress by analyzing worker log files.

    Args:
        log_file: Optional path to specific log file. If not provided, auto-detects latest

    Returns:
        Dictionary with progress statistics and estimates
    """
    # Auto-detect log file if not provided
    if log_file is None:
        # Try to find the most recent embedder log file
        index_path = get_index_path()
        index_path / "embedder_*.log"
        log_files = sorted(index_path.parent.glob("embedder_*.log"))

        if not log_files:
            # Try in the index directory itself
            log_files = sorted(index_path.glob("embedder_*.log"))

        if not log_files:
            logger.error("❌ No log files found! Please provide log file path.")
            return {"error": "No log files found"}

        log_file = log_files[-1]  # Use most recent

    if not log_file.exists():
        logger.error(f"❌ Log file not found: {log_file}")
        return {"error": f"Log file not found: {log_file}"}

    # Read log file
    with log_file.open(encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    # Count successful API calls and errors
    success_calls = [line for line in lines if "200 OK" in line]
    error_lines = [line for line in lines if "ERROR" in line]

    logger.info("=" * 70)
    logger.info(
        f"INDEXING PROGRESS MONITOR - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    logger.info("=" * 70)

    results = {
        "log_file": str(log_file),
        "timestamp": datetime.now().isoformat(),
        "success_calls": len(success_calls),
        "errors": len(error_lines),
        "is_active": False,
    }

    # Parse timing
    if success_calls:
        try:
            first_time = datetime.strptime(
                success_calls[0].split(" - ")[0], "%Y-%m-%d %H:%M:%S,%f"
            )
            last_time = datetime.strptime(
                success_calls[-1].split(" - ")[0], "%Y-%m-%d %H:%M:%S,%f"
            )

            elapsed_seconds = (last_time - first_time).total_seconds()
            elapsed_hours = elapsed_seconds / 3600

            rate_per_hour = (
                len(success_calls) / elapsed_hours if elapsed_hours > 0 else 0
            )

            logger.info("\n📊 ACTIVITY STATUS:")
            logger.info(f"   Started: {first_time.strftime('%H:%M:%S')}")
            logger.info(f"   Latest:  {last_time.strftime('%H:%M:%S')}")
            logger.info(f"   Running: {elapsed_hours:.1f} hours")

            # Check if still active (last call within 2 minutes)
            time_since_last = (datetime.now() - last_time).total_seconds()
            is_active = time_since_last < 120

            if is_active:
                logger.info(
                    f"   Status:  🟢 ACTIVE (last call {int(time_since_last)}s ago)"
                )
            else:
                logger.warning(
                    f"   Status:  🔴 STOPPED (last call {int(time_since_last / 60)} min ago)"
                )

            results["is_active"] = is_active
            results["start_time"] = first_time.isoformat()
            results["last_time"] = last_time.isoformat()
            results["elapsed_hours"] = elapsed_hours
            results["rate_per_hour"] = rate_per_hour
            results["time_since_last_seconds"] = time_since_last

            logger.info("\n📈 PROGRESS:")
            logger.info(f"   API Calls Made: {len(success_calls):,}")
            logger.info(f"   Processing Rate: {rate_per_hour:.1f} calls/hour")
            logger.info(f"   Errors: {len(error_lines)}")

            # Estimate completion
            logger.info("\n⏱️  ESTIMATES:")
            logger.info("   If 1 call = 1 chunk:")
            remaining = 8370  # Approximate remaining chunks
            hours_left = remaining / rate_per_hour if rate_per_hour > 0 else 0
            completion = datetime.now() + timedelta(hours=hours_left)
            logger.info(f"      Remaining: ~{remaining:,} chunks")
            logger.info(
                f"      Time left: ~{hours_left:.1f} hours ({hours_left / 24:.1f} days)"
            )
            logger.info(f"      Complete by: {completion.strftime('%Y-%m-%d %H:%M')}")

            results["estimated_remaining"] = remaining
            results["estimated_hours_left"] = hours_left
            results["estimated_completion"] = completion.isoformat()

        except Exception as e:
            logger.error(f"Error parsing log: {e}")
            results["error"] = f"Error parsing log: {e}"
    else:
        logger.error("❌ No successful API calls found in log")
        results["error"] = "No successful API calls found"

    # Show recent activity
    logger.info("\n📝 LAST 3 LOG ENTRIES:")
    for line in lines[-3:]:
        logger.info(f"   {line.strip()[:100]}")

    logger.info("=" * 70)

    return results


if __name__ == "__main__":
    # Support running different functions from command line
    if len(sys.argv) > 1:
        command = sys.argv[1]

        if command == "analyze":
            analyze_file_processing()
        elif command == "stats":
            root = Path(sys.argv[2]) if len(sys.argv) > 2 else None
            get_file_statistics(root)
        elif command == "chunks":
            export_dir = sys.argv[2] if len(sys.argv) > 2 else "."
            count_total_chunks(export_dir)
        elif command == "monitor":
            log_file = Path(sys.argv[2]) if len(sys.argv) > 2 else None
            monitor_indexing_progress(log_file)
        else:
            print(f"Unknown command: {command}")
            print("Usage: python statistics.py [analyze|stats|chunks|monitor] [args]")
    else:
        print("Usage: python statistics.py [analyze|stats|chunks|monitor] [args]")
