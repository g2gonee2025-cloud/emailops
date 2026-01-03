#!/usr/bin/env python3
"""Download GGUF model using huggingface_hub Python API."""

import os
import shlex
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download
from huggingface_hub.utils import HfHubHTTPError

MODEL_REPO = "mradermacher/KaLM-Embedding-Gemma3-12B-2511-GGUF"
FILENAME = "KaLM-Embedding-Gemma3-12B-2511.Q8_0.gguf"
OUTPUT_DIR = Path("models")


def main():
    """Main function to download and set up the GGUF model."""
    print("=== GGUF Model Downloader (8-bit) ===")
    print(f"Model: {MODEL_REPO}")
    print(f"File: {FILENAME}")
    print(f"Output: {OUTPUT_DIR}")
    print()

    try:
        OUTPUT_DIR.mkdir(exist_ok=True)
    except (OSError, PermissionError) as e:
        print(
            f"Error: Could not create output directory at '{OUTPUT_DIR}': {e}",
            file=sys.stderr,
        )
        sys.exit(1)

    print("Downloading (~12GB) to cache... This may take 10-20 minutes.")
    try:
        # Download to cache to avoid duplicating the large model file.
        path = hf_hub_download(
            repo_id=MODEL_REPO,
            filename=FILENAME,
        )
    except HfHubHTTPError as e:
        print(
            f"Error: Failed to download model '{FILENAME}' from '{MODEL_REPO}': {e}",
            file=sys.stderr,
        )
        print(
            "Please check the model repository and your network connection.",
            file=sys.stderr,
        )
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during download: {e}", file=sys.stderr)
        sys.exit(1)

    print("\n=== Download Complete ===")
    print(f"Model file cached at: {path}")
    try:
        print(f"Size: {os.path.getsize(path) / 1e9:.2f} GB")
    except OSError as e:
        print(f"Warning: Could not get file size for '{path}': {e}", file=sys.stderr)

    # Create or update the symlink to point to the cached model file.
    symlink = OUTPUT_DIR / "kalm-12b-q8.gguf"
    cached_path = Path(path)

    try:
        if symlink.is_symlink():
            # If the symlink is broken or points to the wrong file, update it.
            if (
                not symlink.resolve().exists()
                or symlink.resolve() != cached_path.resolve()
            ):
                print(f"Updating incorrect or broken symlink at '{symlink}'...")
                symlink.unlink()
                symlink.symlink_to(cached_path)
                print(f"Updated symlink: {symlink} -> {cached_path}")
            else:
                print(f"Symlink already exists and is correct: {symlink}")
        elif symlink.exists():
            # A file or directory exists at the path, which is not a symlink.
            print(
                f"Error: A file or directory already exists at '{symlink}' and is not a symlink.",
                file=sys.stderr,
            )
            print("Please remove it and run the script again.", file=sys.stderr)
            sys.exit(1)
        else:
            # The symlink does not exist, so create it.
            symlink.symlink_to(cached_path)
            print(f"Created symlink: {symlink} -> {cached_path}")
    except (OSError, FileExistsError) as e:
        print(
            f"Error: Could not create or update symlink at '{symlink}': {e}",
            file=sys.stderr,
        )
        sys.exit(1)

    quoted_path = shlex.quote(str(symlink.resolve()))
    print("\nTo use the model, set the environment variable:")
    print(f"export OUTLOOKCORTEX_GGUF_MODEL_PATH={quoted_path}")


if __name__ == "__main__":
    main()
