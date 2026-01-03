import hashlib
import json
import logging
import shutil
import sys
from pathlib import Path


def find_repo_root(marker="pyproject.toml"):
    """Searches for the repository root from the script's location."""
    start_dir = Path(__file__).resolve().parent
    for parent in [start_dir, *start_dir.parents]:
        if (parent / marker).exists():
            return parent
    raise FileNotFoundError(f"Could not find repository root with marker '{marker}'")


REPO_ROOT = find_repo_root()
BACKEND_SRC = REPO_ROOT / "backend" / "src"
if str(BACKEND_SRC) not in sys.path:
    sys.path.insert(0, str(BACKEND_SRC))


from cortex.ingestion.conv_manifest.validation import scan_and_refresh


def setup_test_env(root: Path):
    try:
        if root.exists():
            shutil.rmtree(root)
        root.mkdir(parents=True)

        conv_dir = root / "test_conversation"
        conv_dir.mkdir()

        # Create Conversation.txt
        conv_txt = conv_dir / "Conversation.txt"
        content = (
            b"From: test@example.com\nDate: 2025-01-01 10:00:00 +0000\n\nHello world."
        )
        conv_txt.write_bytes(content)

        # Calculate expected hash (normalized LF)
        sha256 = hashlib.sha256()
        sha256.update(content.replace(b"\r\n", b"\n"))
        expected_hash = sha256.hexdigest()

        # Create attachments dir
        (conv_dir / "attachments").mkdir()

        # Create PERFECT manifest that aligns with what scan_and_refresh generates.
        # The `messages` array is the source of truth for participant and date extraction.
        manifest = {
            # Preserved fields (not checked by _manifests_differ but modification would cause a write)
            "smart_subject": "Preserved Subject",
            "conv_id": None,
            "conv_key_type": None,
            "subject_label": "test_conversation",
            # Recalculated and checked fields
            "manifest_version": "1",
            "folder": "test_conversation",
            "sha256_conversation": expected_hash,
            "paths": {
                "conversation_txt": "Conversation.txt",
                "attachments_dir": "attachments/",
            },
            "attachment_count": 0,
            "participants": ["test@example.com"],  # Derived from messages
            "last_from": "test@example.com",  # Derived from messages
            "last_to": [],  # Derived from messages
            # Preserved fields that are also sources for calculation.
            # If these are present, their values are kept.
            "started_at_utc": "2025-01-01T10:00:00Z",
            "ended_at_utc": "2025-01-01T10:00:00Z",
            "message_count": 1,
            # The `messages` array is the source for participant and date extraction logic.
            "messages": [
                {
                    "id": "preserve_me",
                    "from": {"name": "", "smtp": "test@example.com"},
                    "to": [],
                    "cc": [],
                    "date": "2025-01-01T10:00:00Z",
                }
            ],
        }

        (conv_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    except (OSError, PermissionError) as e:
        logging.error(f"Failed to set up test environment at '{root}': {e}")
        raise

    return root


def run_test():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    test_root = Path("./temp_test_validation_idempotency")
    original_error = None

    try:
        setup_test_env(test_root)

        logging.info("Running scan_and_refresh on perfectly valid manifest...")
        try:
            report = scan_and_refresh(test_root)
        except Exception as e:
            logging.error(f"scan_and_refresh failed: {e}")
            original_error = e
            report = None

        if report:
            logging.info(f"Folders Scanned: {report.folders_scanned}")
            logging.info(f"Manifests Created: {report.manifests_created}")
            logging.info(f"Manifests Updated: {report.manifests_updated}")

            if report.manifests_created == 0 and report.manifests_updated == 0:
                logging.info("SUCCESS: No writes occurred for valid manifest.")
            else:
                logging.error("FAILURE: Manifest was written despite being valid.")
                if not original_error:
                    original_error = RuntimeError("Idempotency check failed.")
        else:
            logging.error(
                "FAILURE: Test could not be run because scan_and_refresh failed."
            )
            if not original_error:
                original_error = RuntimeError(
                    "scan_and_refresh returned None unexpectedly."
                )

    except Exception as e:
        logging.error(f"An unexpected error occurred during the test run: {e}")
        original_error = e
    finally:
        if test_root.exists():
            try:
                shutil.rmtree(test_root)
            except (OSError, PermissionError) as e:
                logging.error(f"Cleanup failed for '{test_root}': {e}")
                if not original_error:
                    original_error = e

    if original_error:
        sys.exit(1)


if __name__ == "__main__":
    run_test()
