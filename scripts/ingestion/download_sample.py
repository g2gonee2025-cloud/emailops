import os
import shutil
import subprocess
from pathlib import Path

S3_BUCKET = ":s3:emailops-bucket/Outlook"
DEST_DIR = "temp_validation_s3_20"


def main():
    if Path(DEST_DIR).exists():
        print(f"Cleaning {DEST_DIR}...")
        try:
            shutil.rmtree(DEST_DIR)
        except Exception as e:
            print(f"Warning: could not delete {DEST_DIR}: {e}")

    dest_path = Path(DEST_DIR)
    dest_path.mkdir(parents=True, exist_ok=True)

    with Path("folders_list.txt").open(encoding="utf-8") as f:
        lines = f.readlines()

    parsed_folders = []
    for line in lines:
        if not line.strip():
            continue
        # Extract folder path from rclone lsd output
        # Output format: "          -1 2024-10-30 09:30:10        -1 Folder Name"
        # We assume the first 4 columns are metadata
        parts = line.split(" -1 ", 1)
        if len(parts) == 2:
            folder_name = parts[1].strip()
            # Check for unicode issues or skip
            try:
                folder_name.encode("utf-8")
                parsed_folders.append(folder_name)
            except UnicodeEncodeError:
                print(f"Skipping folder with weird chars: {folder_name}")
        else:
            # Silent skip or log debug
            pass

    print(f"Found {len(parsed_folders)} folders to download.")

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    rclone_bin = "rclone"
    SAMPLE_LIMIT = 20

    print(f"Checking rclone availability: {rclone_bin}")
    try:
        subprocess.run(
            [rclone_bin, "version"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        print("Error: rclone not found in PATH.")
        return

    for i, folder in enumerate(parsed_folders):
        print(f"Downloading {folder}...")

        # We must escape the source path if it has weird chars, but rclone handles it usually if quoted.
        src = f"{S3_BUCKET}/{folder}"
        dst = dest_path / folder

        cmd = [rclone_bin, "copy", src, str(dst), "--transfers", "8"]
        subprocess.run(cmd, check=False)

        if i + 1 >= SAMPLE_LIMIT:
            break

    print("\nDownload complete.")


if __name__ == "__main__":
    main()
