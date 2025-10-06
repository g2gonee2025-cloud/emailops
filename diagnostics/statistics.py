#!/usr/bin/env python3
"""
Consolidated analysis and monitoring functionality for file statistics, chunk counting, and indexing progress.
Consolidates functionality from file_processing_analysis.py, file_stats.py, count_chunks.py, and monitor_indexing.py.
"""

import os
import sys
import pickle
from pathlib import Path
from datetime import datetime, timedelta
from collections import Counter
from typing import Dict, Any, Optional

from diagnostics.utils import setup_logging, get_index_path, get_export_root


# Setup logging
logger = setup_logging()


def analyze_file_processing() -> None:
    """
    Display detailed analysis of which files get chunked vs ignored in EmailOps.
    Shows processing rules and statistics for different file types.
    """
    print("=" * 80)
    print("EMAILOPS FILE PROCESSING ANALYSIS")
    print("=" * 80)

    print("\n📊 YOUR OUTLOOK EXPORT STATISTICS")
    print("-" * 80)
    print("Total Files: 25,024")
    print("Conversations: 3,369")
    print()

    print("✅ FILES THAT GET CHUNKED (Processed & Indexed)")
    print("-" * 80)
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
            print(f"  {ext:12} {count:5,} files - {desc}")
            total_chunked += count

    print(f"\n  TOTAL:      {total_chunked:5,} files ({(total_chunked/25024)*100:.1f}% of all files)")

    print("\n❌ FILES THAT GET IGNORED (Not Processed)")
    print("-" * 80)
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
            print(f"  {ext:12} {count:5,} files - {desc}")
            total_ignored += count

    print(f"\n  TOTAL:      {total_ignored:5,} files ({(total_ignored/25024)*100:.1f}% of all files)")

    print("\n" + "=" * 80)
    print("DETAILED EXTRACTION RULES")
    print("=" * 80)

    print("\n1. CONVERSATION.TXT FILES")
    print("-" * 80)
    print("  Always chunked (primary content)")
    print("  Each conversation -> 1 Conversation.txt")
    print("  Your data: 3,369 files")
    print("  Average chunks per conversation: ~38.3")

    print("\n2. TEXT-BASED FORMATS (Extracted Fully)")
    print("-" * 80)
    print("  .txt, .md, .log, .json, .yaml, .yml, .csv")
    print("  .html, .htm, .xml (tags stripped)")
    print("  Encoding: UTF-8 with Latin-1 fallback")

    print("\n3. MICROSOFT OFFICE FORMATS")
    print("-" * 80)
    print("  .docx (python-docx)")
    print("     - Extracts paragraphs and tables")
    print("     - Your data: 229 files")
    print()
    print("  .doc (pywin32 on Windows, docx2txt fallback)")
    print("     - Requires Microsoft Word on Windows for best results")
    print("     - Your data: 45 files")
    print()
    print("  .xlsx (pandas + openpyxl)")
    print("     - Extracts all sheets as text")
    print("     - Your data: 736 files")
    print()
    print("  .xls (pandas + xlrd)")
    print("     - Extracts all sheets as text")
    print("     - Your data: 76 files")
    print()
    print("  .pptx (NOT SUPPORTED)")
    print("     - Ignored during chunking")
    print("     - Your data: 19 files")

    print("\n4. PDF DOCUMENTS")
    print("-" * 80)
    print("  .pdf (pypdf)")
    print("     - Extracts text from all pages")
    print("     - Skips encrypted PDFs")
    print("     - Your data: 4,167 files (16.7% of all files!)")

    print("\n5. SPECIAL FILES")
    print("-" * 80)
    print("  .json (manifest.json, summary.json)")
    print("     - Used for metadata only, not chunked")
    print("     - Your data: 3,370 files")
    print()
    print("  .log files")
    print("     - System-generated, ignored")
    print("     - Your data: 3,369 files")
    print()
    print("  .zip archives")
    print("     - Not extracted or processed")
    print("     - Your data: 52 files")
    print()
    print("  .eml, .msg, .rpmsg")
    print("     - Email formats not currently supported")
    print("     - Your data: 20 files total")

    print("\n" + "=" * 80)
    print("CHUNKING PROCESS")
    print("=" * 80)

    print("""
1. Each Conversation.txt is ALWAYS processed
2. Attachments in Attachments/ folder are scanned
3. Supported file types have text extracted
4. Files > 8,000 characters are split into chunks
5. Each chunk is 1,600 characters with 200-char overlap
6. Chunks are saved to JSON files in _chunks/chunks/

Your Processing Results:
  - 3,369 conversations processed
  - 12,924 chunk files created
  - 129,055 total chunks
  - Average 38.3 chunks per conversation
  - Average 10 chunks per JSON file
""")

    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"""
PROCESSED: {total_chunked:,} files ({(total_chunked/25024)*100:.1f}%)
  - All Conversation.txt files
  - PDFs (4,167 files - your largest category!)
  - Office documents (1,086 files total)
  - Text/CSV files

IGNORED: {total_ignored:,} files ({(total_ignored/25024)*100:.1f}%)
  - Metadata JSON files
  - Log files
  - Unsupported formats (PowerPoint, archives, .eml)

KEY INSIGHT:
Your system is processing {(total_chunked/25024)*100:.1f}% of files, which is EXCELLENT!
The 4,167 PDF files being chunked is particularly valuable for search.
""")

    print("=" * 80)


def get_file_statistics(root: Optional[Path] = None) -> Dict[str, Any]:
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

    print("=" * 80)
    print("FILE STATISTICS ANALYZER")
    print("=" * 80)

    # Count conversation folders
    convos = [d for d in root.iterdir() if d.is_dir() and not d.name.startswith('_')]
    print(f"\n📁 Conversation Folders: {len(convos):,}")

    # Count Conversation.txt files
    conv_txt = sum(1 for d in convos if (d / 'Conversation.txt').exists())
    print(f"📄 Conversation.txt Files: {conv_txt:,}")

    # Count all files by extension
    print("\n📊 Scanning all files...")
    exts = Counter()
    total_files = 0

    for d in convos:
        for f in d.rglob('*'):
            if f.is_file():
                total_files += 1
                exts[f.suffix.lower() if f.suffix else '(no extension)'] += 1

    print(f"✅ Total Files Scanned: {total_files:,}")

    print("\n📈 Top 20 File Extensions:")
    print("-" * 80)
    for ext, count in exts.most_common(20):
        pct = (count / total_files) * 100 if total_files > 0 else 0
        print(f"  {ext:20} {count:8,} files ({pct:5.1f}%)")

    print("\n" + "=" * 80)

    # Return statistics dictionary
    return {
        "root_path": str(root),
        "conversation_folders": len(convos),
        "conversation_txt_files": conv_txt,
        "total_files": total_files,
        "extensions": dict(exts),
        "top_extensions": dict(exts.most_common(20))
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
        emb_dir = export_path / "_index" / "embeddings"
        total_chunks = 0

        if not emb_dir.is_dir():
            print(f"Error: Embeddings directory not found at {emb_dir}")
            return -1

        pickle_files = sorted(emb_dir.glob("worker_*_batch_*.pkl"))

        if not pickle_files:
            print("No processed batch files found yet.")
            return 0

        for pkl_file in pickle_files:
            try:
                with open(pkl_file, "rb") as f:
                    data = pickle.load(f)
                    # The 'chunks' key holds a list of the actual chunk dictionaries
                    num_chunks_in_batch = len(data.get("chunks", []))
                    total_chunks += num_chunks_in_batch
            except Exception as e:
                print(f"Could not process file {pkl_file}: {e}", file=sys.stderr)
                continue

        print(f"Total chunks created so far: {total_chunks:,}")
        return total_chunks

    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        return -1


def monitor_indexing_progress(log_file: Optional[Path] = None) -> Dict[str, Any]:
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
        log_pattern = index_path / "embedder_*.log"
        log_files = sorted(index_path.parent.glob("embedder_*.log"))
        
        if not log_files:
            # Try in the index directory itself
            log_files = sorted(index_path.glob("embedder_*.log"))
        
        if not log_files:
            print("❌ No log files found! Please provide log file path.")
            return {"error": "No log files found"}
        
        log_file = log_files[-1]  # Use most recent
    
    if not log_file.exists():
        print(f"❌ Log file not found: {log_file}")
        return {"error": f"Log file not found: {log_file}"}
    
    # Read log file
    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
    
    # Count successful API calls and errors
    success_calls = [l for l in lines if '200 OK' in l]
    error_lines = [l for l in lines if 'ERROR' in l]
    
    print("=" * 70)
    print(f"INDEXING PROGRESS MONITOR - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    results = {
        "log_file": str(log_file),
        "timestamp": datetime.now().isoformat(),
        "success_calls": len(success_calls),
        "errors": len(error_lines),
        "is_active": False
    }
    
    # Parse timing
    if success_calls:
        try:
            first_time = datetime.strptime(success_calls[0].split(' - ')[0], '%Y-%m-%d %H:%M:%S,%f')
            last_time = datetime.strptime(success_calls[-1].split(' - ')[0], '%Y-%m-%d %H:%M:%S,%f')
            
            elapsed_seconds = (last_time - first_time).total_seconds()
            elapsed_hours = elapsed_seconds / 3600
            
            rate_per_hour = len(success_calls) / elapsed_hours if elapsed_hours > 0 else 0
            
            print(f"\n📊 ACTIVITY STATUS:")
            print(f"   Started: {first_time.strftime('%H:%M:%S')}")
            print(f"   Latest:  {last_time.strftime('%H:%M:%S')}")
            print(f"   Running: {elapsed_hours:.1f} hours")
            
            # Check if still active (last call within 2 minutes)
            time_since_last = (datetime.now() - last_time).total_seconds()
            is_active = time_since_last < 120
            
            if is_active:
                print(f"   Status:  🟢 ACTIVE (last call {int(time_since_last)}s ago)")
            else:
                print(f"   Status:  🔴 STOPPED (last call {int(time_since_last/60)} min ago)")
            
            results["is_active"] = is_active
            results["start_time"] = first_time.isoformat()
            results["last_time"] = last_time.isoformat()
            results["elapsed_hours"] = elapsed_hours
            results["rate_per_hour"] = rate_per_hour
            results["time_since_last_seconds"] = time_since_last
            
            print(f"\n📈 PROGRESS:")
            print(f"   API Calls Made: {len(success_calls):,}")
            print(f"   Processing Rate: {rate_per_hour:.1f} calls/hour")
            print(f"   Errors: {len(error_lines)}")
            
            # Estimate completion
            print(f"\n⏱️  ESTIMATES:")
            print(f"   If 1 call = 1 chunk:")
            remaining = 8370  # Approximate remaining chunks
            hours_left = remaining / rate_per_hour if rate_per_hour > 0 else 0
            completion = datetime.now() + timedelta(hours=hours_left)
            print(f"      Remaining: ~{remaining:,} chunks")
            print(f"      Time left: ~{hours_left:.1f} hours ({hours_left/24:.1f} days)")
            print(f"      Complete by: {completion.strftime('%Y-%m-%d %H:%M')}")
            
            results["estimated_remaining"] = remaining
            results["estimated_hours_left"] = hours_left
            results["estimated_completion"] = completion.isoformat()
            
        except Exception as e:
            print(f"Error parsing log: {e}")
            results["error"] = f"Error parsing log: {e}"
    else:
        print("❌ No successful API calls found in log")
        results["error"] = "No successful API calls found"
    
    # Show recent activity
    print(f"\n📝 LAST 3 LOG ENTRIES:")
    for line in lines[-3:]:
        print(f"   {line.strip()[:100]}")
    
    print("=" * 70)
    
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
