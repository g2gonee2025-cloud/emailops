#!/usr/bin/env python3
"""Create minimal test structure for GUI testing."""

import json
from pathlib import Path

import numpy as np

# Create export root structure
export_root = Path(r"C:\Users\ASUS\Desktop\Outlook")
export_root.mkdir(parents=True, exist_ok=True)

# Create index directory with minimal files
index_dir = export_root / "_index"
index_dir.mkdir(exist_ok=True)

# Create empty mapping.json
mapping_file = index_dir / "mapping.json"
mapping_file.write_text("[]", encoding="utf-8")

# Create dummy embeddings file
embeddings = np.zeros((0, 768), dtype=np.float32)  # Empty embeddings array
embeddings_file = index_dir / "embeddings.npy"
np.save(embeddings_file, embeddings)

# Create meta.json
meta = {
    "provider": "vertex",
    "model": "gemini-embedding-001",
    "actual_dimensions": 768,
    "index_type": "flat",
    "created_at": "2024-01-01T00:00:00Z",
    "num_folders": 0
}
meta_file = index_dir / "meta.json"
meta_file.write_text(json.dumps(meta, indent=2), encoding="utf-8")

# Create chunks directory
chunks_dir = export_root / "_chunks"
chunks_dir.mkdir(exist_ok=True)

# Create at least one test conversation directory
test_conv = export_root / "test_conversation_001"
test_conv.mkdir(exist_ok=True)

# Create a minimal Conversation.txt
conv_file = test_conv / "Conversation.txt"
conv_file.write_text("""From: test@example.com
To: user@example.com
Subject: Test Email
Date: 2024-01-01T00:00:00Z

This is a test email for GUI development.
""", encoding="utf-8")

print(f"Created test structure in {export_root}")
print(f"  - Index directory: {index_dir}")
print(f"  - Chunks directory: {chunks_dir}")
print(f"  - Test conversation: {test_conv}")
