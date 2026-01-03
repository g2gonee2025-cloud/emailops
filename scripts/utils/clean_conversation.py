"""
DEPRECATED: This script's logic has been merged into the Core module.

Use `cortex.email_processing.clean_email_text()` instead.

This file will be removed in a future release. The patterns and logic have been
consolidated into `backend/src/cortex/email_processing.py` to avoid duplication.

For local file cleaning, use the cortex CLI instead.
"""

import re
import shutil
import sys
import warnings
from pathlib import Path
from typing import List, Union

try:
    # Tkinter is optional; we handle environments without a GUI.
    import tkinter as tk
    from tkinter import filedialog
except ImportError:  # Tkinter is not installed
    tk = None
    filedialog = None

URL_FULLMATCH_RE = re.compile(r"(?:https?://\S+)", re.IGNORECASE)


BOILERPLATE_KEYWORDS = [
    # Common noisy markers / disclaimers / system notes (extend as needed)
    "Email from external sender",
    "Email from External Sender",
    "Classification : Interne",
    "Classification : Internal",
    "Classification : Public",
    "Ce message, et ses éventuelles pièces jointes",
    "Ce message a été classifié",
    "L'Internet ne permettant pas d'assurer l'intégrité de ce Message",
    "Go Green, Avoid Printing",
    "This message, and its attachments",
    "This communication is confidential",
    "DISCLAIMER:",
    "legal notice",
    "confidentiality notice",
    "Please consider the environment before printing",
    "A member of the Nasco Insurance Group",
]

# Lines that are usually just web / marketing noise
DOMAIN_NOISE = [
    "CHALHOUBGROUP.COM",
    "CAREERS.CHALHOUBGROUP.COM",
]

# Standard length used when normalizing long separator lines
SEPARATOR_LINE_LENGTH = 40

HEADER_PATTERNS = [
    # Date + From header at top of each email in Conversation.txt
    re.compile(r"^\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}\s+\|\s+From:", re.IGNORECASE),
    re.compile(r"^From:", re.IGNORECASE),
    re.compile(r"^To:", re.IGNORECASE),
    re.compile(r"^Cc:", re.IGNORECASE),
    re.compile(r"^Subject:", re.IGNORECASE),
    re.compile(r"^主题:", re.IGNORECASE),
]

SEPARATOR_RE = re.compile(r"^[-_=]{5,}$")


def looks_like_header(line: str) -> bool:
    """Heuristic: is this an email header / metadata line?"""
    return any(pat.search(line) for pat in HEADER_PATTERNS)


def is_boilerplate(line: str) -> bool:
    """Return True if line looks like disclaimer / footer noise."""
    stripped = line.strip()
    if not stripped:
        return False

    # Single-domain / URL lines
    if URL_FULLMATCH_RE.fullmatch(stripped):
        return True
    if stripped.upper() in DOMAIN_NOISE:
        return True
    if re.fullmatch(r"www\.[A-Za-z0-9\.-]+", stripped):
        return True

    # Contains any known boilerplate keyword
    lower = stripped.lower()
    for key in BOILERPLATE_KEYWORDS:
        if key.lower() in lower:
            return True

    # Very long address-ish lines (lots of commas, no @) - usually postal/signature blocks
    return bool(len(stripped) > 80 and stripped.count(",") >= 3 and "@" not in stripped)


def normalize_and_filter_lines(text: str) -> str:
    """Perform whitespace normalization and boilerplate removal in one pass."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = text.split("\n")
    cleaned_lines = []

    for line in lines:
        # First, check if the raw line is boilerplate.
        if is_boilerplate(line):
            continue

        # If not, then process it.
        line = line.strip()

        if line and not looks_like_header(line):
            line = re.sub(r"\s+", " ", line)

        if SEPARATOR_RE.match(line):
            line = "-" * SEPARATOR_LINE_LENGTH

        cleaned_lines.append(line)

    # Now, collapse blank lines from the already-filtered list.
    final_lines = []
    blank_count = 0
    for line in cleaned_lines:
        if not line:
            blank_count += 1
            if blank_count > 1:
                continue
        else:
            blank_count = 0
        final_lines.append(line)

    return "\n".join(final_lines).strip() + "\n"


def merge_wrapped_lines(text: str) -> str:
    """Merge artificially wrapped lines into paragraphs for better RAG chunks.

    Heuristic:
    - keep header lines and separator lines on their own;
    - inside body blocks, join consecutive non-empty, non-header, non-bullet lines.
    """
    lines = text.split("\n")
    merged = []
    buffer = ""

    def flush_buffer():
        nonlocal buffer
        if buffer:
            merged.append(buffer.strip())
            buffer = ""

    for line in lines:
        stripped = line.strip()

        # Preserve blank lines as paragraph separators
        if stripped == "":
            flush_buffer()
            merged.append("")  # keep single blank line
            continue

        if looks_like_header(stripped) or SEPARATOR_RE.match(stripped):
            flush_buffer()
            merged.append(stripped)
            continue

        # Bullet / numbered list / dash lines: start new buffer
        if re.match(r"^([\-•●*]|\d+[\.\)])\s+", stripped):
            flush_buffer()
            merged.append(stripped)
            continue

        # Otherwise, this is body text: join into current paragraph
        if buffer:
            buffer += " " + stripped
        else:
            buffer = stripped

    flush_buffer()
    return "\n".join(merged)


def clean_conversation_text(raw: str) -> str:
    """Full cleaning pipeline for a Conversation.txt file."""
    # 1) Normalize whitespace and remove boilerplate in a single pass
    text = normalize_and_filter_lines(raw)

    # 2) Merge wrapped lines inside paragraphs to make embedding-friendly text
    text = merge_wrapped_lines(text)

    # 4) Final pass to ensure we don't have excessive blank lines
    lines = text.split("\n")
    final_lines = []
    blank_count = 0
    for line in lines:
        if line.strip() == "":
            blank_count += 1
            if blank_count > 1:
                continue
            final_lines.append("")
        else:
            blank_count = 0
            final_lines.append(line.rstrip())

    return "\n".join(final_lines).strip() + "\n"


def choose_root_directory() -> str:
    """Let the user pick the root directory.

    Priority:
    1) Command-line argument (if provided);
    2) GUI folder picker via Tkinter (if available);
    3) Text prompt fallback.
    """
    # 1) CLI arg
    if len(sys.argv) > 1:
        candidate = Path(sys.argv[1]).expanduser()
        if candidate.is_dir():
            resolved = candidate.resolve()
            print(f"Using root directory from command line: {resolved}")
            return str(resolved)
        else:
            print(f"Path from command line is not a directory: {candidate}")

    # 2) GUI folder picker
    if tk is not None and filedialog is not None:
        root = None
        try:
            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            selected = filedialog.askdirectory(
                title="Select ROOT directory (folder that contains conversation subfolders)"
            )
            if selected:
                selected_path = Path(selected)
                if selected_path.is_dir():
                    resolved = selected_path.resolve()
                    print(f"Selected root directory: {resolved}")
                    return str(resolved)
        except Exception as e:  # TclError or others
            print(
                f"GUI directory picker not available ({e}). Falling back to console input."
            )
        finally:
            if root:
                root.destroy()

    # 3) Console input fallback
    while True:
        try:
            path = (
                input(
                    "Enter path to root directory (folder that contains conversation subfolders): "
                )
                .strip()
                .strip('"')
                .strip("'")
            )
            candidate = Path(path).expanduser()
            if candidate.is_dir():
                return str(candidate.resolve())
            print("That path does not exist or is not a directory. Please try again.")
        except (KeyboardInterrupt, EOFError):
            print("\nOperation cancelled by user.")
            sys.exit(1)


def find_conversation_files(root_dir: Union[str, Path]) -> List[Path]:
    """Return list of Conversation.txt files at:
    - root_dir/Conversation.txt (if present)
    - root_dir/*/Conversation.txt (1 level of subfolders only)
    """
    root_path = Path(root_dir)
    results: List[Path] = []

    # Root-level Conversation.txt
    root_file = root_path / "Conversation.txt"
    if root_file.is_file():
        results.append(root_file)

    # One level down: immediate subdirectories only
    try:
        for entry in root_path.iterdir():
            if entry.is_dir():
                candidate = entry / "Conversation.txt"
                if candidate.is_file():
                    results.append(candidate)
    except (PermissionError, OSError) as e:
        print(f"[WARN] Could not scan subdirectories in {root_path}: {e}", file=sys.stderr)

    return results


def process_file(path: Path, dry_run: bool = False) -> None:
    """Clean a single Conversation.txt file in place, with a .bak backup."""
    print(f"\n--- Processing: {path}")
    try:
        # Use errors='replace' to avoid silent data loss of undecodable characters
        with path.open("r", encoding="utf-8", errors="replace") as f:
            raw = f.read()
    except Exception as e:
        print(f"  [ERROR] Failed to read file: {e}")
        return

    original_size = len(raw)

    cleaned = clean_conversation_text(raw)
    cleaned_size = len(cleaned)

    print(f"  Original length: {original_size} chars")
    print(f"  Cleaned  length: {cleaned_size} chars")

    if dry_run:
        print("  Dry-run mode: NOT writing changes.")
        return

    # Backup original (Conversation.txt.bak) once
    backup_path = path.with_suffix(path.suffix + ".bak")
    if not backup_path.exists():
        try:
            shutil.copy2(path, backup_path)
            print(f"  Backup created: {backup_path}")
        except Exception as e:
            print(f"  [WARN] Could not create backup: {e}")

    try:
        with path.open("w", encoding="utf-8", errors="replace") as f:
            f.write(cleaned)
        print("  [OK] File cleaned and overwritten.")
    except Exception as e:
        print(f"  [ERROR] Failed to write cleaned file: {e}")


def main():
    warnings.warn(
        "clean_conversation.py is deprecated and will be removed in a future release. "
        "The core logic has been moved to cortex.email_processing.clean_email_text().",
        DeprecationWarning,
        stacklevel=2,
    )
    root_dir = choose_root_directory()
    print(f"\nRoot directory: {root_dir}")

    root_path = Path(root_dir)
    files = find_conversation_files(root_path)
    if not files:
        print("No Conversation.txt files found (1-level deep). Nothing to do.")
        return

    print(f"Found {len(files)} Conversation.txt file(s):")
    for p in files:
        print(" -", p)

    # If you want a dry run first, set dry_run=True below.
    dry_run = False

    for path in files:
        process_file(path, dry_run=dry_run)

    print("\nDone. All Conversation.txt files cleaned.")


if __name__ == "__main__":
    main()
