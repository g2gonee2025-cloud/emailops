#!/usr/bin/env python
"""
Simple script to run the Outlook exporter with proper parameters.
"""

import sys
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from emailops.outlook_exporter.exporter import OutlookExporter


def main():
    # Configuration from .env
    output_root = Path(r"C:\Users\ASUS\Desktop\Outlook")
    folders = [
        r"\\Outlook Data File\\Inbox",
        r"\\Outlook Data File\\Sent Items"
    ]

    print("Starting Outlook export...")
    print(f"Output directory: {output_root}")
    print(f"Folders to export: {folders}")

    try:
        # Create exporter instance
        exporter = OutlookExporter(
            output_root=output_root,
            outlook_profile=None,  # Use default profile
            enforce_offline=False   # Don't require offline mode for now
        )

        # Run the export
        print("\nStarting export process...")
        exporter.export_folders(
            folder_paths=folders,
            full_export=False,  # Use incremental export
            since_utc=None     # Will use state from previous runs if available
        )

        print("\nExport completed successfully!")

    except Exception as e:
        print(f"\nError during export: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())
