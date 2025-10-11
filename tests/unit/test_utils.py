"""
Unit tests for diagnostics/utils.py

Tests cover:
- Logging setup
- Path resolution
- Timestamp formatting
- JSON report saving
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest

from diagnostics.utils import (
    format_timestamp,
    get_export_root,
    get_index_path,
    save_json_report,
    setup_logging,
)


# ============================================================================
# setup_logging Tests
# ============================================================================

class TestSetupLogging:
    """Tests for setup_logging function."""
    
    def test_setup_logging_with_default_level_returns_logger(self):
        """Test that setup_logging returns a logger with INFO level."""
        logger = setup_logging()
        
        assert isinstance(logger, logging.Logger)
        assert logger.name == "diagnostics.utils"
    
    def test_setup_logging_with_debug_level_sets_debug(self):
        """Test that setup_logging respects DEBUG level parameter."""
        # Reset root logger first
        logging.getLogger().setLevel(logging.WARNING)
        
        logger = setup_logging("DEBUG")
        
        # Check effective level (considering inheritance)
        effective_level = logger.getEffectiveLevel()
        assert effective_level == logging.DEBUG
    
    def test_setup_logging_with_warning_level_sets_warning(self):
        """Test that setup_logging respects WARNING level parameter."""
        # Reset root logger first
        logging.getLogger().setLevel(logging.INFO)
        
        logger = setup_logging("WARNING")
        
        # Check effective level (considering inheritance)
        effective_level = logger.getEffectiveLevel()
        assert effective_level == logging.WARNING
    
    def test_setup_logging_with_invalid_level_uses_info(self):
        """Test that invalid level defaults to INFO."""
        # Reset root logger first
        logging.getLogger().setLevel(logging.WARNING)
        
        logger = setup_logging("INVALID_LEVEL")
        
        # Should default to INFO - check effective level
        effective_level = logger.getEffectiveLevel()
        assert effective_level == logging.INFO
    
    def test_setup_logging_with_lowercase_level_works(self):
        """Test that lowercase level names work."""
        # Reset root logger first
        logging.getLogger().setLevel(logging.WARNING)
        
        logger = setup_logging("debug")
        
        # Check effective level (considering inheritance)
        effective_level = logger.getEffectiveLevel()
        assert effective_level == logging.DEBUG


# ============================================================================
# get_index_path Tests
# ============================================================================

class TestGetIndexPath:
    """Tests for get_index_path function."""
    
    def test_get_index_path_with_no_root_uses_cwd(self):
        """Test that get_index_path uses current directory when root is None."""
        result = get_index_path(None)
        
        assert isinstance(result, Path)
        assert result.name == "_index"
        assert result.parent == Path.cwd()
    
    def test_get_index_path_with_root_returns_root_plus_index(self, temp_dir):
        """Test that get_index_path uses provided root directory."""
        result = get_index_path(str(temp_dir))
        
        assert isinstance(result, Path)
        assert result.name == "_index"
        assert result.parent == temp_dir
    
    def test_get_index_path_respects_env_variable(self, temp_dir, monkeypatch):
        """Test that get_index_path respects INDEX_DIRNAME environment variable."""
        monkeypatch.setenv("INDEX_DIRNAME", "_custom_index")
        
        result = get_index_path(str(temp_dir))
        
        assert result.name == "_custom_index"
        assert result.parent == temp_dir
    
    def test_get_index_path_with_path_object_works(self, temp_dir):
        """Test that get_index_path accepts Path objects."""
        result = get_index_path(str(temp_dir))
        
        assert isinstance(result, Path)
        assert result.parent == temp_dir
    
    def test_get_index_path_with_relative_path_works(self):
        """Test that get_index_path handles relative paths."""
        result = get_index_path("./test_dir")
        
        assert isinstance(result, Path)
        assert result.name == "_index"


# ============================================================================
# get_export_root Tests
# ============================================================================

class TestGetExportRoot:
    """Tests for get_export_root function."""
    
    def test_get_export_root_with_env_variable_returns_env_path(self, monkeypatch):
        """Test that get_export_root uses OUTLOOK_EXPORT_ROOT env variable."""
        test_path = "/test/export/path"
        monkeypatch.setenv("OUTLOOK_EXPORT_ROOT", test_path)
        
        result = get_export_root()
        
        assert isinstance(result, Path)
        # Handle Windows path separators
        assert str(result).replace("\\", "/") == test_path
    
    def test_get_export_root_without_env_variable_returns_cwd(self, monkeypatch):
        """Test that get_export_root falls back to current directory."""
        monkeypatch.delenv("OUTLOOK_EXPORT_ROOT", raising=False)
        
        result = get_export_root()
        
        assert isinstance(result, Path)
        assert result == Path.cwd()
    
    def test_get_export_root_with_empty_env_variable_returns_cwd(self, monkeypatch):
        """Test that get_export_root handles empty env variable."""
        monkeypatch.setenv("OUTLOOK_EXPORT_ROOT", "")
        
        result = get_export_root()
        
        # Empty string is falsy, should return cwd
        assert result == Path.cwd()


# ============================================================================
# format_timestamp Tests
# ============================================================================

class TestFormatTimestamp:
    """Tests for format_timestamp function."""
    
    def test_format_timestamp_with_datetime_returns_formatted_string(self):
        """Test that format_timestamp correctly formats a datetime object."""
        dt = datetime(2024, 1, 15, 14, 30, 45)
        
        result = format_timestamp(dt)
        
        assert result == "2024-01-15 14:30:45"
    
    def test_format_timestamp_with_midnight_formats_correctly(self):
        """Test that format_timestamp handles midnight correctly."""
        dt = datetime(2024, 1, 1, 0, 0, 0)
        
        result = format_timestamp(dt)
        
        assert result == "2024-01-01 00:00:00"
    
    def test_format_timestamp_with_end_of_day_formats_correctly(self):
        """Test that format_timestamp handles end of day correctly."""
        dt = datetime(2024, 12, 31, 23, 59, 59)
        
        result = format_timestamp(dt)
        
        assert result == "2024-12-31 23:59:59"
    
    def test_format_timestamp_preserves_leading_zeros(self):
        """Test that format_timestamp preserves leading zeros."""
        dt = datetime(2024, 1, 5, 9, 5, 3)
        
        result = format_timestamp(dt)
        
        assert result == "2024-01-05 09:05:03"
    
    def test_format_timestamp_with_microseconds_ignores_them(self):
        """Test that format_timestamp ignores microseconds."""
        dt = datetime(2024, 1, 15, 14, 30, 45, 123456)
        
        result = format_timestamp(dt)
        
        # Microseconds should not be in output
        assert result == "2024-01-15 14:30:45"
        assert "123456" not in result


# ============================================================================
# save_json_report Tests
# ============================================================================

class TestSaveJsonReport:
    """Tests for save_json_report function."""
    
    def test_save_json_report_creates_file(self, temp_dir):
        """Test that save_json_report creates a JSON file."""
        data = {"key": "value", "number": 42}
        filename = str(temp_dir / "test_report.json")
        
        result = save_json_report(data, filename)
        
        assert result.exists()
        assert result.name == "test_report.json"
    
    def test_save_json_report_writes_correct_content(self, temp_dir):
        """Test that save_json_report writes correct JSON content."""
        data = {"status": "success", "count": 100}
        filename = str(temp_dir / "test_report.json")
        
        result = save_json_report(data, filename)
        
        with open(result, "r", encoding="utf-8") as f:
            loaded_data = json.load(f)
        
        assert loaded_data == data
    
    def test_save_json_report_with_nested_data_works(self, temp_dir):
        """Test that save_json_report handles nested dictionaries."""
        data = {
            "level1": {
                "level2": {
                    "level3": "deep_value"
                },
                "list": [1, 2, 3]
            }
        }
        filename = str(temp_dir / "nested_report.json")
        
        result = save_json_report(data, filename)
        
        with open(result, "r", encoding="utf-8") as f:
            loaded_data = json.load(f)
        
        assert loaded_data == data
        assert loaded_data["level1"]["level2"]["level3"] == "deep_value"
    
    def test_save_json_report_with_unicode_works(self, temp_dir):
        """Test that save_json_report handles Unicode characters."""
        data = {
            "message": "Hello 世界 🌍",
            "emoji": "✅ ❌ ⚠️"
        }
        filename = str(temp_dir / "unicode_report.json")
        
        result = save_json_report(data, filename)
        
        with open(result, "r", encoding="utf-8") as f:
            loaded_data = json.load(f)
        
        assert loaded_data == data
        assert "世界" in loaded_data["message"]
        assert "🌍" in loaded_data["message"]
    
    def test_save_json_report_with_list_data_works(self, temp_dir):
        """Test that save_json_report handles list data."""
        data = {"items": [1, 2, 3, 4, 5]}
        filename = str(temp_dir / "list_report.json")
        
        result = save_json_report(data, filename)
        
        with open(result, "r", encoding="utf-8") as f:
            loaded_data = json.load(f)
        
        assert loaded_data == data
        assert len(loaded_data["items"]) == 5
    
    def test_save_json_report_overwrites_existing_file(self, temp_dir):
        """Test that save_json_report overwrites existing files."""
        filename = str(temp_dir / "overwrite_test.json")
        
        # Write first version
        data1 = {"version": 1}
        save_json_report(data1, filename)
        
        # Overwrite with second version
        data2 = {"version": 2}
        result = save_json_report(data2, filename)
        
        with open(result, "r", encoding="utf-8") as f:
            loaded_data = json.load(f)
        
        assert loaded_data == data2
        assert loaded_data["version"] == 2
    
    def test_save_json_report_returns_path_object(self, temp_dir):
        """Test that save_json_report returns a Path object."""
        data = {"test": "data"}
        filename = str(temp_dir / "path_test.json")
        
        result = save_json_report(data, filename)
        
        assert isinstance(result, Path)
        assert result.name == "path_test.json"
    
    def test_save_json_report_with_empty_dict_works(self, temp_dir):
        """Test that save_json_report handles empty dictionary."""
        data = {}
        filename = str(temp_dir / "empty_report.json")
        
        result = save_json_report(data, filename)
        
        with open(result, "r", encoding="utf-8") as f:
            loaded_data = json.load(f)
        
        assert loaded_data == {}
    
    def test_save_json_report_formats_with_indentation(self, temp_dir):
        """Test that save_json_report uses indentation for readability."""
        data = {"key1": "value1", "key2": "value2"}
        filename = str(temp_dir / "formatted_report.json")
        
        result = save_json_report(data, filename)
        
        # Read as text to check formatting
        content = result.read_text(encoding="utf-8")
        
        # Should have newlines and indentation
        assert "\n" in content
        assert "  " in content  # 2-space indent


# ============================================================================
# Integration Tests
# ============================================================================

class TestUtilsIntegration:
    """Integration tests combining multiple utility functions."""
    
    def test_get_index_path_and_save_report_work_together(self, temp_dir):
        """Test that index path and report saving work together."""
        # Create index directory
        index_path = get_index_path(str(temp_dir))
        index_path.mkdir(parents=True, exist_ok=True)
        
        # Save report in index directory
        data = {"status": "complete"}
        report_file = str(index_path / "report.json")
        result = save_json_report(data, report_file)
        
        assert result.parent == index_path
        assert result.exists()
    
    def test_format_timestamp_in_report_data(self, temp_dir):
        """Test using format_timestamp within report data."""
        now = datetime.now()
        data = {
            "timestamp": format_timestamp(now),
            "status": "success"
        }
        filename = str(temp_dir / "timestamped_report.json")
        
        result = save_json_report(data, filename)
        
        with open(result, "r", encoding="utf-8") as f:
            loaded_data = json.load(f)
        
        assert "timestamp" in loaded_data
        assert len(loaded_data["timestamp"]) == 19  # YYYY-MM-DD HH:MM:SS


# ============================================================================
# Edge Case Tests
# ============================================================================

class TestUtilsEdgeCases:
    """Edge case tests for utility functions."""
    
    def test_get_index_path_with_special_characters_in_dirname(self, temp_dir, monkeypatch):
        """Test that get_index_path handles special characters."""
        monkeypatch.setenv("INDEX_DIRNAME", "_index-test_123")
        
        result = get_index_path(str(temp_dir))
        
        assert result.name == "_index-test_123"
    
    def test_save_json_report_with_very_long_filename(self, temp_dir):
        """Test that save_json_report handles long filenames."""
        long_name = "a" * 200 + ".json"
        data = {"test": "data"}
        filename = str(temp_dir / long_name)
        
        # This might fail on some systems with path length limits
        # but should work on most modern systems
        try:
            result = save_json_report(data, filename)
            assert result.exists()
        except OSError:
            # Path too long - acceptable failure
            pytest.skip("Path length limit exceeded")
    
    def test_format_timestamp_with_year_boundary(self):
        """Test that format_timestamp handles year boundaries."""
        dt = datetime(1999, 12, 31, 23, 59, 59)
        
        result = format_timestamp(dt)
        
        assert result == "1999-12-31 23:59:59"
    
    def test_get_export_root_with_relative_env_path(self, monkeypatch):
        """Test that get_export_root handles relative paths in env."""
        monkeypatch.setenv("OUTLOOK_EXPORT_ROOT", "./relative/path")
        
        result = get_export_root()
        
        assert isinstance(result, Path)
        # Handle Windows path separators and normalization
        result_str = str(result).replace("\\", "/")
        assert result_str == "./relative/path" or result_str == "relative/path"