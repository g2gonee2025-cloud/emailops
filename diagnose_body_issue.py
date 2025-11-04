"""Diagnostic script to check why email bodies are empty."""
import win32com.client

outlook = win32com.client.Dispatch("Outlook.Application")
ns = outlook.GetNamespace("MAPI")

# Get Inbox
folder = ns.GetDefaultFolder(6)  # olFolderInbox

print("Checking first 5 emails in Inbox:\n")
print("=" * 80)

items = folder.Items
items.Sort("[ReceivedTime]", True)  # Descending

count = 0
for item in items:
    if count >= 5:
        break

    try:
        if item.Class != 43:  # Not a mail item
            continue

        print(f"\nEmail {count + 1}:")
        print(f"  Subject: {item.Subject[:60] if item.Subject else 'N/A'}")
        print(f"  DownloadState: {item.DownloadState} (1=Full, 0=Header)")
        print(f"  Size: {item.Size} bytes")
        print(f"  BodyFormat: {item.BodyFormat} (1=Plain, 2=HTML, 3=RTF)")

        # Try Body
        body = item.Body if hasattr(item, 'Body') else None
        body_len = len(body) if body else 0
        print(f"  Body length: {body_len}")
        if body_len > 0:
            print(f"  Body preview: {body[:100]}")

        # Try HTMLBody
        html_body = item.HTMLBody if hasattr(item, 'HTMLBody') else None
        html_len = len(html_body) if html_body else 0
        print(f"  HTMLBody length: {html_len}")
        if html_len > 0:
            print(f"  HTMLBody preview: {html_body[:100]}")

        count += 1
    except Exception as e:
        print(f"  ERROR: {e}")
        continue

print("\n" + "=" * 80)
print(f"\nChecked {count} emails")
print("\nDiagnosis:")
print("- If DownloadState == 1 but bodies are empty: Bodies not yet synced")
print("- If DownloadState == 0: Still in headers-only mode despite setting")
print("- If DownloadState == 1 and bodies present: Code issue")
