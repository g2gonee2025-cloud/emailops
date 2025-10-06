#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analysis of which files get chunked vs ignored in EmailOps.
Based on your actual Outlook export statistics.
"""

print("=" * 80)
print("EMAILOPS FILE PROCESSING ANALYSIS")
print("=" * 80)

print("\n?? YOUR OUTLOOK EXPORT STATISTICS")
print("-" * 80)
print("Total Files: 25,024")
print("Conversations: 3,369")
print()

print("? FILES THAT GET CHUNKED (Processed & Indexed)")
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

print("\n? FILES THAT GET IGNORED (Not Processed)")
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
