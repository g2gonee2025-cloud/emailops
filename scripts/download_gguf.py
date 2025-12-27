#!/usr/bin/env python3
"""Download GGUF model using huggingface_hub Python API."""

import os
from pathlib import Path

from huggingface_hub import hf_hub_download

MODEL_REPO = "mradermacher/KaLM-Embedding-Gemma3-12B-2511-GGUF"
FILENAME = "KaLM-Embedding-Gemma3-12B-2511.Q8_0.gguf"
OUTPUT_DIR = Path("models")

print("=== GGUF Model Downloader (8-bit) ===")
print(f"Model: {MODEL_REPO}")
print(f"File: {FILENAME}")
print(f"Output: {OUTPUT_DIR}")
print()

OUTPUT_DIR.mkdir(exist_ok=True)

print("Downloading (~12GB)... This may take 10-20 minutes.")
path = hf_hub_download(
    repo_id=MODEL_REPO,
    filename=FILENAME,
    local_dir=OUTPUT_DIR,
    local_dir_use_symlinks=False,
)

print("\n=== Download Complete ===")
print(f"Model file: {path}")
print(f"Size: {os.path.getsize(path) / 1e9:.2f} GB")

# Create simple symlink
symlink = OUTPUT_DIR / "kalm-12b-q8.gguf"
if not symlink.exists():
    symlink.symlink_to(Path(path).name)
    print(f"Created symlink: {symlink}")

print(f"\nSet env: export OUTLOOKCORTEX_GGUF_MODEL_PATH={symlink}")
