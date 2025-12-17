import hashlib
import json
import logging
import shutil
import sys
from pathlib import Path

# Add backend/src to path so imports work
sys.path.append("/root/workspace/emailops-vertex-ai/backend/src")

from cortex.ingestion.conv_manifest.validation import scan_and_refresh


def setup_test_env(root: Path):
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)

    conv_dir = root / "test_conversation"
    conv_dir.mkdir()

    # Create Conversation.txt
    conv_txt = conv_dir / "Conversation.txt"
    content = b"From: test@example.com\nDate: 2025-01-01 10:00:00\n\nHello world."
    conv_txt.write_bytes(content)

    # Calculate expected hash (normalized LF)
    sha256 = hashlib.sha256()
    sha256.update(content.replace(b"\r\n", b"\n"))
    expected_hash = sha256.hexdigest()

    # Create attachments dir
    (conv_dir / "attachments").mkdir()

    # Create PERFECT manifest
    manifest = {
        "manifest_version": "1",
        "folder": "test_conversation",
        "subject_label": "test_conversation",
        "message_count": 0,
        "started_at_utc": "2025-01-01T10:00:00Z",  # Matches file content
        "ended_at_utc": "2025-01-01T10:00:00Z",
        "attachment_count": 0,
        "paths": {
            "conversation_txt": "Conversation.txt",
            "attachments_dir": "attachments/",
        },
        "sha256_conversation": expected_hash,
        "messages": [{"id": "preserve_me"}],  # Critical field to preserve
        "smart_subject": "Preserved Subject",  # Critical field to preserve
    }

    (conv_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    return root


def run_test():
    logging.basicConfig(level=logging.INFO)
    test_root = Path("./temp_test_validation_idempotency")

    try:
        setup_test_env(test_root)

        print("Running scan_and_refresh on perfectly valid manifest...")
        report = scan_and_refresh(test_root)

        print(f"Folders Scanned: {report.folders_scanned}")
        print(f"Manifests Created: {report.manifests_created}")
        print(f"Manifests Updated: {report.manifests_updated}")

        if report.manifests_created == 0 and report.manifests_updated == 0:
            print("\nSUCCESS: No writes occurred for valid manifest.")
        else:
            print("\nFAILURE: Manifest was written despite being valid.")

    finally:
        if test_root.exists():
            shutil.rmtree(test_root)


if __name__ == "__main__":
    run_test()
