from pathlib import Path
from unittest.mock import patch

import pytest
from cortex.orchestration.nodes import _safe_stat_mb


@pytest.fixture
def temp_file(tmp_path: Path):
    """Create a temporary file for testing."""
    file_path = tmp_path / "test_file.txt"
    file_path.write_text("hello")
    return file_path


async def test_safe_stat_mb_existing_file(temp_file: Path):
    """Test _safe_stat_mb with an existing file."""
    expected_size_mb = 5 / (1024 * 1024)
    assert _safe_stat_mb(temp_file) == pytest.approx(expected_size_mb)


async def test_safe_stat_mb_non_existent_file(tmp_path: Path):
    """Test _safe_stat_mb with a non-existent file."""
    non_existent_file = tmp_path / "non_existent.txt"
    assert _safe_stat_mb(non_existent_file) == pytest.approx(0.0)


async def test_safe_stat_mb_zero_byte_file(tmp_path: Path):
    """Test _safe_stat_mb with a zero-byte file."""
    zero_byte_file = tmp_path / "zero.txt"
    zero_byte_file.touch()
    assert _safe_stat_mb(zero_byte_file) == pytest.approx(0.0)


@patch("pathlib.Path.stat")
async def test_safe_stat_mb_stat_raises_exception(mock_stat, tmp_path: Path):
    """Test _safe_stat_mb when stat() raises an exception."""
    mock_stat.side_effect = OSError("Permission denied")
    file_path = tmp_path / "some_file.txt"
    file_path.touch()  # The file needs to exist for stat to be called
    assert _safe_stat_mb(file_path) == pytest.approx(0.0)
