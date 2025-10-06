#!/usr/bin/env python3
"""Quick file statistics for Outlook directory."""
from pathlib import Path
from collections import Counter
import sys

root = Path("C:/Users/ASUS/Desktop/Outlook")

print("=" * 80)
print("FILE STATISTICS ANALYZER")
print("=" * 80)

# Count conversation folders
convos = [d for d in root.iterdir() if d.is_dir() and not d.name.startswith('_')]
print(f"\n?? Conversation Folders: {len(convos):,}")

# Count Conversation.txt files
conv_txt = sum(1 for d in convos if (d / 'Conversation.txt').exists())
print(f"?? Conversation.txt Files: {conv_txt:,}")

# Count all files by extension
print("\n?? Scanning all files...")
exts = Counter()
total_files = 0

for d in convos:
    for f in d.rglob('*'):
        if f.is_file():
            total_files += 1
            exts[f.suffix.lower() if f.suffix else '(no extension)'] += 1

print(f"? Total Files Scanned: {total_files:,}")

print("\n?? Top 20 File Extensions:")
print("-" * 80)
for ext, count in exts.most_common(20):
    pct = (count / total_files) * 100 if total_files > 0 else 0
    print(f"  {ext:20} {count:8,} files ({pct:5.1f}%)")

print("\n" + "=" * 80)
