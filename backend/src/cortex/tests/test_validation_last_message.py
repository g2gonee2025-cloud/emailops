# -*- coding: utf-8 -*-
"""Tests for validation manifest last_from/last_to extraction"""

import json
import sys
from pathlib import Path

import pytest

# Ensure the project src directory is on the import path
project_root = Path(__file__).resolve().parents[2]
src_path = project_root
sys.path.append(str(src_path))

from cortex.ingestion.conv_manifest.validation import scan_and_refresh  # noqa: E402


@pytest.fixture
def temp_convo_dir(tmp_path: Path):
    # Create a conversation folder structure
    conv_dir = tmp_path / "conv_folder"
    conv_dir.mkdir()
    # Write Conversation.txt with two messages
    conv_txt = conv_dir / "Conversation.txt"
    conv_txt.write_text(
        "2024-10-07 14:43 | From: alice@example.com | To: bob@example.com; carol@example.com\n"
        "2024-10-07 15:50 | From: dave@example.com | To: eve@example.com\n",
        encoding="utf-8",
    )
    # Create empty attachments directory (required by validator)
    (conv_dir / "attachments").mkdir()
    return conv_dir


def test_last_message_fields_are_written(temp_convo_dir: Path):
    # Run the validator on the temporary conversation directory
    report = scan_and_refresh(temp_convo_dir.parent)
    # The validator should have created a manifest.json inside the folder
    manifest_path = temp_convo_dir / "manifest.json"
    assert manifest_path.exists(), "manifest.json was not created"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    # Verify that the last message's From and To fields are present
    assert manifest.get("last_from") == "dave@example.com"
    # The To list should contain the single recipient from the last line
    assert manifest.get("last_to") == ["eve@example.com"]
    # Ensure participants still contain all unique participants from both messages
    expected_participants = sorted(
        [
            "alice@example.com",
            "bob@example.com",
            "carol@example.com",
            "dave@example.com",
            "eve@example.com",
        ]
    )
    assert sorted(manifest.get("participants", [])) == expected_participants
    # Clean up temporary files (pytest's tmp_path handles this automatically)
    # Ensure the report indicates a manifest was created
    assert any(p.issue == "manifest_written:created" for p in report.problems)
