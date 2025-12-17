import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
BACKEND_SRC = (REPO_ROOT / "backend" / "src").resolve()
if str(BACKEND_SRC) not in sys.path:
    sys.path.insert(0, str(BACKEND_SRC))


# Import necessary logic re-implemented or imported to inspect the diff
def _calculate_conversation_hash(path: Path) -> str:
    sha256 = hashlib.sha256()
    with path.open("rb") as f:
        content = f.read()
        content = content.replace(b"\r\n", b"\n")
        sha256.update(content)
    return sha256.hexdigest()


def analyze_diffs():
    root = Path("temp_s3_validation")
    if not root.exists():
        print("Temp dir not found. Did you run verify_s3_manifests.py?")
        return

    print("Scanning for differences...")
    diff_count = 0

    for conv_dir in root.iterdir():
        if not conv_dir.is_dir():
            continue

        manifest_path = conv_dir / "manifest.json"
        conv_txt = conv_dir / "Conversation.txt"

        if not manifest_path.exists() or not conv_txt.exists():
            continue

        try:
            old = json.loads(manifest_path.read_text())
        except Exception:
            continue

        # Calculate expected values
        try:
            expected_hash = _calculate_conversation_hash(conv_txt)
        except Exception:
            continue

        attachments_dir = conv_dir / "attachments"
        expected_att_count = 0
        if attachments_dir.exists():
            expected_att_count = len(list(attachments_dir.iterdir()))

        # Compare
        reasons = []

        # 1. Hash
        if old.get("sha256_conversation") != expected_hash:
            reasons.append(
                f"Hash Mismatch (Old: {old.get('sha256_conversation')[:8]}... New: {expected_hash[:8]}...)"
            )

        # 2. Attachment Count
        if old.get("attachment_count") != expected_att_count:
            reasons.append(
                f"Attachment Count (Old: {old.get('attachment_count')} New: {expected_att_count})"
            )

        # 3. Version
        if old.get("manifest_version") != "1":
            reasons.append(
                f"Version Upgrade (Old: {old.get('manifest_version')} New: 1)"
            )

        # 4. Folder name mismatch (sometimes happens if renamed)
        if old.get("folder") != conv_dir.name:
            reasons.append(
                f"Folder Name (Old: {old.get('folder')} New: {conv_dir.name})"
            )

        if reasons:
            diff_count += 1
            print(f"\n[{conv_dir.name}]")
            for r in reasons:
                print(f"  - {r}")

    print(f"\nTotal differences found: {diff_count}")


if __name__ == "__main__":
    analyze_diffs()
