import hashlib
import json
import logging
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
BACKEND_SRC = (REPO_ROOT / "backend" / "src").resolve()
if str(BACKEND_SRC) not in sys.path:
    sys.path.insert(0, str(BACKEND_SRC))

from cortex.ingestion.conv_manifest.validation import scan_and_refresh  # noqa: E402


def setup_mass_env(root: Path, count: int = 200):
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)

    print(f"Generating {count} synthetic conversation folders...")

    for i in range(count):
        folder_name = f"conversation_{i:03d}"
        conv_dir = root / folder_name
        conv_dir.mkdir()

        # Create Conversation.txt
        conv_txt = conv_dir / "Conversation.txt"
        content = f"From: user{i}@example.com\nDate: 2025-01-01 10:00:00\n\nMessage {i}".encode()
        conv_txt.write_bytes(content)

        # Create attachments dir
        (conv_dir / "attachments").mkdir()

        # Calculate expected hash
        sha256 = hashlib.sha256()
        sha256.update(content.replace(b"\r\n", b"\n"))
        expected_hash = sha256.hexdigest()

        # Create RICH manifest (simulating S3 source)
        manifest = {
            "manifest_version": "1",
            "folder": folder_name,
            "subject_label": f"Subject {i}",
            "message_count": 1,
            "started_at_utc": "2025-01-01T10:00:00Z",
            "ended_at_utc": "2025-01-01T10:00:00Z",
            "attachment_count": 0,
            "paths": {
                "conversation_txt": "Conversation.txt",
                "attachments_dir": "attachments/",
            },
            "sha256_conversation": expected_hash,
            # CRITICAL FIELDS TO PRESERVE
            "messages": [
                {
                    "from": {"name": f"User {i}", "smtp": f"user{i}@example.com"},
                    "date": "2025-01-01T10:00:00Z",
                    "subject": f"Subject {i}",
                }
            ],
            "smart_subject": f"Smart Subject {i}",
            "participants": [{"name": f"User {i}", "role": "client"}],
        }

        (conv_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    return root


def run_test():
    logging.basicConfig(level=logging.ERROR)  # Quiet logs
    test_root = Path("./temp_mass_validation")

    try:
        setup_mass_env(test_root, 200)

        print("\nRunning scan_and_refresh on 200 folders...")
        report = scan_and_refresh(test_root)

        print("\n=== Validation Report Results ===")
        print(f"Folders Scanned: {report.folders_scanned}")
        print(f"Manifests Created: {report.manifests_created}")
        print(f"Manifests Updated: {report.manifests_updated}")
        print(f"Problems Found: {len(report.problems)}")

        # Verify preservation
        print("\n=== Verifying Data Preservation ===")
        preserved_count = 0
        failed_count = 0

        for i in range(200):
            folder_name = f"conversation_{i:03d}"
            manifest_path = test_root / folder_name / "manifest.json"

            try:
                data = json.loads(manifest_path.read_text())

                # Check critical fields
                has_messages = "messages" in data and len(data["messages"]) == 1
                has_smart_subject = "smart_subject" in data
                has_participants = "participants" in data

                if has_messages and has_smart_subject and has_participants:
                    preserved_count += 1
                else:
                    failed_count += 1
                    print(f"FAILED: {folder_name} missing fields!")
            except Exception as e:
                print(f"ERROR reading {folder_name}: {e}")
                failed_count += 1

        print(f"Preserved Manifests: {preserved_count}/200")
        print(f"Failed Manifests:    {failed_count}/200")

        if failed_count == 0:
            print("\nSUCCESS: All 200 manifests preserved rich data.")
        else:
            print("\nFAILURE: Some manifests lost data.")

    finally:
        if test_root.exists():
            shutil.rmtree(test_root)


if __name__ == "__main__":
    run_test()
