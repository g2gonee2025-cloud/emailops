"""Unit tests for cortex.utils.atomic_io module."""

import json
from pathlib import Path

import pytest
from cortex.utils.atomic_io import atomic_write_json, atomic_write_json_async


class TestAtomicWriteJson:
    def test_basic_json_write(self, tmp_path):
        """Test writing a simple JSON object."""
        file_path = "test.json"
        data = {"key": "value", "number": 42}

        atomic_write_json(tmp_path, file_path, data)

        target = tmp_path / file_path
        assert target.exists()
        with open(target) as f:
            written = json.load(f)
        assert written == data

    def test_creates_parent_directory(self, tmp_path):
        """Test that parent directories are created if needed."""
        file_path = Path("nested/deep/test.json")
        data = {"nested": True}

        atomic_write_json(tmp_path, file_path, data)

        target = tmp_path / file_path
        assert target.exists()
        with open(target) as f:
            written = json.load(f)
        assert written == data

    def test_overwrites_existing_file(self, tmp_path):
        """Test that existing file is replaced atomically."""
        file_path = "test.json"

        # Write initial
        atomic_write_json(tmp_path, file_path, {"version": 1})

        # Overwrite
        atomic_write_json(tmp_path, file_path, {"version": 2})

        target = tmp_path / file_path
        with open(target) as f:
            written = json.load(f)
        assert written["version"] == 2

    def test_unicode_data(self, tmp_path):
        """Test handling of unicode data."""
        file_path = "unicode.json"
        data = {"greeting": "Hello, 世界! 🌍"}

        atomic_write_json(tmp_path, file_path, data)

        target = tmp_path / file_path
        with open(target, encoding="utf-8") as f:
            written = json.load(f)
        assert written["greeting"] == data["greeting"]

    def test_complex_nested_structure(self, tmp_path):
        """Test writing complex nested JSON."""
        file_path = "complex.json"
        data = {
            "array": [1, 2, 3],
            "nested": {"a": {"b": {"c": True}}},
            "nullval": None,
        }

        atomic_write_json(tmp_path, file_path, data)

        target = tmp_path / file_path
        with open(target) as f:
            written = json.load(f)
        assert written == data

    def test_path_traversal_blocked(self, tmp_path):
        """Test that writing outside the base directory is blocked."""
        # Attempt to write to the parent of tmp_path
        file_path = "../traversal.json"
        data = {"attack": True}

        with pytest.raises(PermissionError):
            atomic_write_json(tmp_path, file_path, data)

        assert not (tmp_path / file_path).resolve().exists()


@pytest.mark.asyncio
class TestAtomicWriteJsonAsync:
    async def test_async_basic_json_write(self, tmp_path):
        """Test async writing of a simple JSON object."""
        file_path = "test_async.json"
        data = {"async": True}

        await atomic_write_json_async(tmp_path, file_path, data)

        target = tmp_path / file_path
        assert target.exists()
        with open(target) as f:
            written = json.load(f)
        assert written == data

    async def test_async_path_traversal_blocked(self, tmp_path):
        """Test that async writing outside the base directory is blocked."""
        file_path = "../traversal_async.json"
        data = {"attack": True}

        with pytest.raises(PermissionError):
            await atomic_write_json_async(tmp_path, file_path, data)

        assert not (tmp_path / file_path).resolve().exists()
