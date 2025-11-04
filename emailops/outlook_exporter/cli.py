from __future__ import annotations

import argparse
import logging
from datetime import UTC, datetime
from pathlib import Path

from .exporter import OutlookExporter


def _parse_since(value: str) -> datetime:
    """
    Parse a flexible --since value into a UTC-aware datetime.
    Accepted examples:
      2025-01-31
      2025-01-31T14:30
      2025-01-31T14:30:00
      2025-01-31T14:30Z
      2025-01-31T14:30:00+02:00
    """
    s = value.strip()
    try:
        if "T" not in s:
            dt = datetime.fromisoformat(f"{s}T00:00:00")
        else:
            if s.endswith(("Z", "z")):
                s = s[:-1] + "+00:00"
            dt = datetime.fromisoformat(s)
    except ValueError as e:
        raise argparse.ArgumentTypeError(
            f"Invalid --since '{value}'. Try formats like "
            f"YYYY-MM-DD, YYYY-MM-DDTHH:MMZ, or YYYY-MM-DDTHH:MM:SS+HH:MM."
        ) from e
    if dt.tzinfo is None:
        dt = dt.astimezone()
    return dt.astimezone(UTC)


def _pick_folders_gui():
    """Use Outlook's native GUI folder picker dialog"""
    import win32com.client
    app = win32com.client.Dispatch("Outlook.Application")
    ns = app.GetNamespace("MAPI")

    selected_folders = []

    print("\nOutlook folder picker will open...")
    print("Select folders one at a time (click OK after each)")
    print("When done selecting all folders, click Cancel\n")

    while True:
        # Show Outlook's native folder picker dialog
        folder = ns.PickFolder()

        if folder is None:
            # User clicked Cancel - done selecting
            break

        # Get folder path
        try:
            folder_path = folder.FolderPath
            if folder_path and folder_path not in selected_folders:
                selected_folders.append(folder_path)
                try:
                    count = folder.Items.Count if hasattr(folder.Items, "Count") else 0
                except Exception:  # Outlook COM can raise generic COM errors
                    count = 0
                print(f"Selected: {folder_path} ({count} items)")
            elif folder_path in selected_folders:
                print(f"Already selected: {folder_path}")
        except Exception as e:
            print(f"Error getting folder path: {e}")
            continue

    return selected_folders


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Export Outlook conversations to EmailOps manifest format."
    )
    ap.add_argument("--output", help="Output root directory (uses .env OUTLOOK_EXPORT_DIR if not specified)")
    ap.add_argument(
        "--folders",
        nargs="+",
        help=r"One or more Outlook folder paths, e.g. \\Mailbox\\Inbox. Use --pick-folders for interactive selection.",
    )
    ap.add_argument(
        "--pick-folders",
        action="store_true",
        help="Interactive folder selection"
    )
    ap.add_argument(
        "--full", action="store_true", help="Full export (ignore previous state)"
    )
    ap.add_argument(
        "--since",
        help="Only export items received since this time (ISO 8601; accepts Z or offsets)",
    )
    ap.add_argument("--profile", help="Outlook profile name (optional)")
    ap.add_argument(
        "--no-offline-check",
        action="store_true",
        help="Do not require Outlook to be in Work Offline mode",
    )
    ap.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    ap.add_argument("--log-file", help="Write logs to file (in addition to console)")
    args = ap.parse_args(argv)

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    log_format = "%(asctime)s %(levelname)s %(name)s: %(message)s"
    
    handlers = [logging.StreamHandler()]  # Always log to console
    
    if args.log_file:
        # Add file handler
        file_handler = logging.FileHandler(args.log_file, encoding='utf-8')
        file_handler.setFormatter(logging.Formatter(log_format))
        handlers.append(file_handler)
    
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=handlers,
        force=True  # Override any existing configuration
    )

    # Handle output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        import os

        from dotenv import load_dotenv
        load_dotenv()
        output_dir_str = os.getenv('OUTLOOK_EXPORT_DIR', './outlook_exports')
        output_dir = Path(output_dir_str)
        print(f"Using output directory from .env: {output_dir.absolute()}")

    since_dt = None
    if args.since:
        try:
            since_dt = _parse_since(args.since)
        except argparse.ArgumentTypeError as e:
            ap.error(str(e))

    # Handle folder selection
    if args.pick_folders:
        selected_folders = _pick_folders_gui()

        if not selected_folders:
            print("\nNo folders selected (cancelled)")
            return 0

        print(f"\n{'='*80}")
        print(f"SELECTED {len(selected_folders)} FOLDERS FOR EXPORT:")
        print('='*80)
        for path in selected_folders:
            print(f"  {path}")
        print('='*80 + "\n")

        folders_to_export = selected_folders
    elif args.folders:
        bad = [p for p in args.folders if not p or not p.startswith("\\")]
        if bad:
            ap.error("Folder paths must start with '\\'. Invalid: " + ", ".join(bad))
        folders_to_export = args.folders
    else:
        ap.error("Either --folders or --pick-folders is required")

    exp = OutlookExporter(
        output_dir,
        outlook_profile=args.profile,
        enforce_offline=(not args.no_offline_check),
    )
    exp.export_folders(folders_to_export, full_export=args.full, since_utc=since_dt)

if __name__ == "__main__":
    raise SystemExit(main())
