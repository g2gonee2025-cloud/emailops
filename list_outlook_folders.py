#!/usr/bin/env python
"""
Script to list available Outlook folders
"""

import sys

try:
    import win32com.client
except ImportError:
    print("pywin32 is required. Install with: pip install pywin32")
    sys.exit(1)

def list_outlook_folders():
    """List all available Outlook folders"""
    try:
        # Connect to Outlook
        outlook = win32com.client.Dispatch("Outlook.Application")
        namespace = outlook.GetNamespace("MAPI")

        print("Available Outlook Folders:")
        print("=" * 50)

        # List all top-level folders (mailboxes/stores)
        for i in range(1, namespace.Folders.Count + 1):
            try:
                folder = namespace.Folders.Item(i)
                print(f"\n\\\\{folder.Name}")

                # List subfolders
                for j in range(1, folder.Folders.Count + 1):
                    try:
                        subfolder = folder.Folders.Item(j)
                        print(f"  \\\\{folder.Name}\\{subfolder.Name}")

                        # Show if it has items
                        try:
                            count = subfolder.Items.Count
                            if count > 0:
                                print(f"    ({count} items)")
                        except Exception:
                            pass

                    except Exception:
                        pass

            except Exception:
                pass

        print("\n" + "=" * 50)
        print("\nNote: Use the folder paths shown above in your export script.")
        print("For example: '\\\\Your Mailbox Name\\\\Inbox'")

    except Exception as e:
        print(f"Error connecting to Outlook: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    list_outlook_folders()
