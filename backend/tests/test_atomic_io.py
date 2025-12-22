"""Unit tests for cortex.utils.atomic_io module."""

import json

from cortex.utils.atomic_io import atomic_write_json


class TestAtomicWriteJson:
    def test_basic_json_write(self, tmp_path):
        """Test writing a simple JSON object."""
        target = tmp_path / "test.json"
        data = {"key": "value", "number": 42}

        atomic_write_json(target, data)

        assert target.exists()
        with open(target) as f:
            written = json.load(f)
        assert written == data

    def test_creates_parent_directory(self, tmp_path):
        """Test that parent directories are created if needed."""
        target = tmp_path / "nested" / "deep" / "test.json"
        data = {"nested": True}

        atomic_write_json(target, data)

        assert target.exists()
        with open(target) as f:
            written = json.load(f)
        assert written == data

    def test_overwrites_existing_file(self, tmp_path):
        """Test that existing file is replaced atomically."""
        target = tmp_path / "test.json"

        # Write initial
        atomic_write_json(target, {"version": 1})

        # Overwrite
        atomic_write_json(target, {"version": 2})

        with open(target) as f:
            written = json.load(f)
        assert written["version"] == 2

    def test_unicode_data(self, tmp_path):
        """Test handling of unicode data."""
        target = tmp_path / "unicode.json"
        data = {"greeting": "Hello, 世界! 🌍"}

        atomic_write_json(target, data)

        with open(target, encoding="utf-8") as f:
            written = json.load(f)
        assert written["greeting"] == data["greeting"]

    def test_complex_nested_structure(self, tmp_path):
        """Test writing complex nested JSON."""
        target = tmp_path / "complex.json"
        data = {
            "array": [1, 2, 3],
            "nested": {"a": {"b": {"c": True}}},
            "nullval": None,
        }

        atomic_write_json(target, data)

        with open(target) as f:
            written = json.load(f)
        assert written == data
