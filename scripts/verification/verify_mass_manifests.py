import hashlib
import json
import logging
import shutil
import sys
from pathlib import Path

# --- Path setup ---
# This script is designed to be run from the repository root.
# It modifies sys.path to allow imports from the backend source code.
try:
    _repo_root = Path(__file__).resolve().parent.parent.parent
    _backend_src = (_repo_root / "backend" / "src").resolve(strict=True)
    if str(_backend_src) not in sys.path:
        sys.path.insert(0, str(_backend_src))
except FileNotFoundError:
    print("ERROR: Could not find the backend source directory.", file=sys.stderr)
    print("Please run this script from the repository root.", file=sys.stderr)
    sys.exit(1)


from cortex.ingestion.conv_manifest.validation import (  # noqa: E402
    ManifestValidationReport,
    scan_and_refresh,
)


def setup_mass_env(root: Path, count: int = 200):
    # SECURITY: Prevent accidental deletion of arbitrary directories.
    resolved_root = root.resolve()
    resolved_cwd = Path.cwd().resolve()
    if not resolved_root.is_relative_to(resolved_cwd) or "temp" not in resolved_root.name:
        print(f"ERROR: Unsafe path provided for deletion: {root}")
        print("To prevent accidental data loss, this script can only operate on a temporary directory within the current working directory.")
        sys.exit(1)
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

        (conv_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )

    return root


def run_test():
    """Set up the test environment, run the scan, and verify the results."""
    logging.basicConfig(level=logging.ERROR)  # Quiet logs
    test_root = Path("./temp_mass_validation")
    conversation_count = 200

    try:
        setup_mass_env(test_root, conversation_count)

        print(f"\nRunning scan_and_refresh on {conversation_count} folders...")
        report: ManifestValidationReport = scan_and_refresh(test_root)

        print("\n=== Validation Report Results ===")
        print(f"Folders Scanned: {report.folders_scanned}")
        print(f"Manifests Created: {report.manifests_created}")
        print(f"Manifests Updated: {report.manifests_updated}")
        print(f"Problems Found: {len(report.problems)}")

        # Verify preservation
        print("\n=== Verifying Data Preservation ===")
        preserved_count = 0
        failed_count = 0

        for i in range(conversation_count):
            folder_name = f"conversation_{i:03d}"
            manifest_path = test_root / folder_name / "manifest.json"

            try:
                data = json.loads(manifest_path.read_text(encoding="utf-8"))

                # Check critical fields
                has_messages = "messages" in data and len(data["messages"]) == 1
                has_smart_subject = "smart_subject" in data
                has_participants = "participants" in data

                if has_messages and has_smart_subject and has_participants:
                    preserved_count += 1
                else:
                    failed_count += 1
                    print(f"FAILED: {folder_name} missing fields!")
            except (json.JSONDecodeError, IOError) as e:
                print(f"ERROR reading {folder_name}: {e}")
                failed_count += 1

        print(f"Preserved Manifests: {preserved_count}/{conversation_count}")
        print(f"Failed Manifests:    {failed_count}/{conversation_count}")

        if failed_count == 0:
            print(f"\nSUCCESS: All {conversation_count} manifests preserved rich data.")
        else:
            print("\nFAILURE: Some manifests lost data.")

    finally:
        if test_root.exists():
            shutil.rmtree(test_root)


if __name__ == "__main__":
    run_test()
