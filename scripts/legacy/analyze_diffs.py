import hashlib
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


# Import necessary logic re-implemented or imported to inspect the diff
def _calculate_conversation_hash(path: Path) -> str:
    sha256 = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(8192):
            sha256.update(chunk.replace(b"\r\n", b"\n"))
    return sha256.hexdigest()


def analyze_diffs():
    root = REPO_ROOT / "temp_s3_validation"
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
        except json.JSONDecodeError:
            print(f"Warning: Could not decode JSON from {manifest_path}")
            continue

        # Calculate expected values
        try:
            expected_hash = _calculate_conversation_hash(conv_txt)
        except OSError as e:
            print(f"Warning: Could not read {conv_txt}: {e}")
            continue

        attachments_dir = conv_dir / "attachments"
        expected_att_count = 0
        if attachments_dir.exists() and attachments_dir.is_dir():
            expected_att_count = sum(
                1 for item in attachments_dir.iterdir() if item.is_file()
            )

        # Compare
        reasons = []

        # 1. Hash
        old_hash = old.get("sha256_conversation")
        if old_hash != expected_hash:
            old_hash_short = f"{old_hash[:8]}..." if old_hash else "None"
            reasons.append(
                f"Hash Mismatch (Old: {old_hash_short} New: {expected_hash[:8]}...)"
            )

        # 2. Attachment Count
        if str(old.get("attachment_count")) != str(expected_att_count):
            reasons.append(
                f"Attachment Count (Old: {old.get('attachment_count')} New: {expected_att_count})"
            )

        # 3. Version
        if str(old.get("manifest_version")) != "1":
            reasons.append(
                f"Version Mismatch (Old: {old.get('manifest_version')} New: 1)"
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
