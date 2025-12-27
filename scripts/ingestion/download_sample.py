import os
import shutil
import subprocess
from pathlib import Path

S3_BUCKET = os.environ.get("S3_BUCKET", ":s3:emailops-bucket/Outlook")
DEST_DIR = os.environ.get("DEST_DIR", "temp_validation_s3_20")


def is_valid_dir_name(name: str) -> bool:
    """Basic validation to prevent path traversal."""
    return ".." not in name and not name.startswith("/")


def main():
    if Path(DEST_DIR).exists():
        print(f"Cleaning {DEST_DIR}...")
        try:
            shutil.rmtree(DEST_DIR)
        except Exception as e:
            print(f"Warning: could not delete {DEST_DIR}: {e}")

    dest_path = Path(DEST_DIR)
    dest_path.mkdir(parents=True, exist_ok=True)

    try:
        with Path("folders_list.txt").open(encoding="utf-8") as f:
            lines = f.readlines()
    except FileNotFoundError:
        print("Error: folders_list.txt not found.")
        return

    parsed_folders = []
    for line in lines:
        if not line.strip():
            continue
        parts = line.split(" -1 ", 1)
        if len(parts) == 2:
            folder_name = parts[1].strip()
            if is_valid_dir_name(folder_name):
                try:
                    folder_name.encode("utf-8")
                    parsed_folders.append(folder_name)
                except UnicodeEncodeError:
                    print(f"Skipping folder with non-UTF-8 chars: {folder_name!r}")
            else:
                print(f"Skipping potentially malicious folder name: {folder_name!r}")
        else:
            pass  # Silent skip

    print(f"Found {len(parsed_folders)} valid folders to download.")

    rclone_bin = "rclone"
    SAMPLE_LIMIT = 20

    print(f"Checking rclone availability: {rclone_bin}")
    try:
        subprocess.run(
            [rclone_bin, "version"],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
    except FileNotFoundError:
        print("Error: rclone not found in PATH.")
        return
    except subprocess.CalledProcessError as e:
        print(f"Error checking rclone version: {e.stderr}")
        return

    for i, folder in enumerate(parsed_folders[:SAMPLE_LIMIT]):
        print(f"Downloading '{folder}'...")

        src = f"{S3_BUCKET}/{folder}"
        dst = dest_path / folder

        try:
            # Use a list of args for security
            cmd = [rclone_bin, "copy", src, str(dst), "--transfers", "8"]
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
            )
            if result.stdout:
                print(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"Error downloading {folder}: {e.stderr}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

    print("\nDownload complete.")


if __name__ == "__main__":
    main()
