"""
Unit tests for diagnostics/statistics.py

Tests cover:
- File statistics analysis
- Chunk counting
- Progress monitoring
- File processing analysis
"""

import pickle
from datetime import datetime

import pytest

from diagnostics.statistics import (
    analyze_file_processing,
    count_total_chunks,
    get_file_statistics,
    monitor_indexing_progress,
)

# ============================================================================
# analyze_file_processing Tests
# ============================================================================

class TestAnalyzeFileProcessing:
    """Tests for analyze_file_processing function."""

    def test_analyze_file_processing_runs_without_error(self, caplog):
        """Test that analyze_file_processing executes successfully."""
        import logging
        with caplog.at_level(logging.INFO):
            analyze_file_processing()

        # Check log messages instead of stdout
        log_text = "\n".join(record.message for record in caplog.records)
        assert "EMAILOPS FILE PROCESSING ANALYSIS" in log_text

    def test_analyze_file_processing_shows_chunked_files(self, caplog):
        """Test that analyze_file_processing displays chunked file types."""
        import logging
        with caplog.at_level(logging.INFO):
            analyze_file_processing()

        log_text = "\n".join(record.message for record in caplog.records)
        assert ".txt" in log_text
        assert ".pdf" in log_text
        assert ".docx" in log_text

    def test_analyze_file_processing_shows_ignored_files(self, caplog):
        """Test that analyze_file_processing displays ignored file types."""
        import logging
        with caplog.at_level(logging.INFO):
            analyze_file_processing()

        log_text = "\n".join(record.message for record in caplog.records)
        assert ".json" in log_text
        assert ".log" in log_text

    def test_analyze_file_processing_shows_summary(self, caplog):
        """Test that analyze_file_processing includes summary section."""
        import logging
        with caplog.at_level(logging.INFO):
            analyze_file_processing()

        log_text = "\n".join(record.message for record in caplog.records)
        assert "SUMMARY" in log_text
        assert "PROCESSED" in log_text
        assert "IGNORED" in log_text


# ============================================================================
# get_file_statistics Tests
# ============================================================================

class TestGetFileStatistics:
    """Tests for get_file_statistics function."""

    def test_get_file_statistics_with_empty_directory(self, temp_dir):
        """Test get_file_statistics with empty directory."""
        result = get_file_statistics(temp_dir)

        assert result["conversation_folders"] == 0
        assert result["conversation_txt_files"] == 0
        assert result["total_files"] == 0

    def test_get_file_statistics_counts_conversations(self, mock_conversation_structure):
        """Test that get_file_statistics counts conversation folders."""
        result = get_file_statistics(mock_conversation_structure.parent)

        assert result["conversation_folders"] >= 1
        assert result["conversation_txt_files"] >= 1

    def test_get_file_statistics_counts_files_by_extension(self, temp_dir):
        """Test that get_file_statistics counts files by extension."""
        # Create test conversation structure
        conv_dir = temp_dir / "conversation_1"
        conv_dir.mkdir()

        # Create various file types
        (conv_dir / "Conversation.txt").touch()
        (conv_dir / "doc.pdf").touch()
        (conv_dir / "data.json").touch()
        (conv_dir / "notes.md").touch()

        result = get_file_statistics(temp_dir)

        assert result["total_files"] == 4
        assert ".txt" in result["extensions"]
        assert ".pdf" in result["extensions"]
        assert ".json" in result["extensions"]
        assert ".md" in result["extensions"]

    def test_get_file_statistics_returns_top_extensions(self, temp_dir):
        """Test that get_file_statistics returns top extensions."""
        conv_dir = temp_dir / "conversation_1"
        conv_dir.mkdir()

        # Create files
        for i in range(5):
            (conv_dir / f"file{i}.txt").touch()
        (conv_dir / "doc.pdf").touch()

        result = get_file_statistics(temp_dir)

        assert "top_extensions" in result
        assert isinstance(result["top_extensions"], dict)
        assert result["top_extensions"][".txt"] == 5

    def test_get_file_statistics_with_no_extension_files(self, temp_dir):
        """Test that get_file_statistics handles files without extensions."""
        conv_dir = temp_dir / "conversation_1"
        conv_dir.mkdir()

        (conv_dir / "Conversation.txt").touch()
        (conv_dir / "README").touch()  # No extension
        (conv_dir / "LICENSE").touch()  # No extension

        result = get_file_statistics(temp_dir)

        assert "(no extension)" in result["extensions"]
        assert result["extensions"]["(no extension)"] == 2

    def test_get_file_statistics_includes_path_info(self, temp_dir):
        """Test that get_file_statistics includes root path in results."""
        result = get_file_statistics(temp_dir)

        assert "root_path" in result
        assert str(temp_dir) in result["root_path"]

    def test_get_file_statistics_handles_nested_directories(self, temp_dir):
        """Test that get_file_statistics searches recursively."""
        conv_dir = temp_dir / "conversation_1"
        conv_dir.mkdir()

        # Create nested structure
        attach_dir = conv_dir / "Attachments"
        attach_dir.mkdir()

        (conv_dir / "Conversation.txt").touch()
        (attach_dir / "doc.pdf").touch()
        (attach_dir / "data.xlsx").touch()

        result = get_file_statistics(temp_dir)

        assert result["total_files"] == 3


# ============================================================================
# count_total_chunks Tests
# ============================================================================

class TestCountTotalChunks:
    """Tests for count_total_chunks function."""

    def test_count_total_chunks_with_missing_directory(self, temp_dir):
        """Test count_total_chunks with missing embeddings directory."""
        result = count_total_chunks(str(temp_dir))

        assert result == -1

    def test_count_total_chunks_with_no_files(self, temp_dir):
        """Test count_total_chunks with empty embeddings directory."""
        emb_dir = temp_dir / "_index" / "embeddings"
        emb_dir.mkdir(parents=True)

        result = count_total_chunks(str(temp_dir))

        assert result == 0

    def test_count_total_chunks_counts_pickle_files(self, temp_dir):
        """Test that count_total_chunks counts chunks in pickle files."""
        emb_dir = temp_dir / "_index" / "embeddings"
        emb_dir.mkdir(parents=True)

        # Create sample pickle file
        chunks_data = {
            "chunks": [
                {"text": "chunk 1"},
                {"text": "chunk 2"},
                {"text": "chunk 3"}
            ]
        }

        pkl_file = emb_dir / "worker_0_batch_00000.pkl"
        with open(pkl_file, "wb") as f:
            pickle.dump(chunks_data, f)

        result = count_total_chunks(str(temp_dir))

        assert result == 3

    def test_count_total_chunks_sums_multiple_files(self, temp_dir):
        """Test that count_total_chunks sums across multiple files."""
        emb_dir = temp_dir / "_index" / "embeddings"
        emb_dir.mkdir(parents=True)

        # Create multiple pickle files
        for i in range(3):
            chunks_data = {
                "chunks": [{"text": f"chunk {j}"} for j in range(5)]
            }
            pkl_file = emb_dir / f"worker_0_batch_{i:05d}.pkl"
            with open(pkl_file, "wb") as f:
                pickle.dump(chunks_data, f)

        result = count_total_chunks(str(temp_dir))

        assert result == 15  # 3 files × 5 chunks

    def test_count_total_chunks_handles_corrupted_pickle(self, temp_dir):
        """Test that count_total_chunks handles corrupted pickle files."""
        emb_dir = temp_dir / "_index" / "embeddings"
        emb_dir.mkdir(parents=True)

        # Create valid pickle
        valid_data = {"chunks": [{"text": "valid chunk"}]}
        valid_file = emb_dir / "worker_0_batch_00000.pkl"
        with open(valid_file, "wb") as f:
            pickle.dump(valid_data, f)

        # Create corrupted pickle
        corrupted_file = emb_dir / "worker_0_batch_00001.pkl"
        corrupted_file.write_bytes(b"corrupted data")

        result = count_total_chunks(str(temp_dir))

        # Should count valid file, skip corrupted
        assert result == 1

    def test_count_total_chunks_handles_missing_chunks_key(self, temp_dir):
        """Test that count_total_chunks handles missing chunks key."""
        emb_dir = temp_dir / "_index" / "embeddings"
        emb_dir.mkdir(parents=True)

        # Create pickle without chunks key
        invalid_data = {"embeddings": "some data"}
        pkl_file = emb_dir / "worker_0_batch_00000.pkl"
        with open(pkl_file, "wb") as f:
            pickle.dump(invalid_data, f)

        result = count_total_chunks(str(temp_dir))

        assert result == 0


# ============================================================================
# monitor_indexing_progress Tests
# ============================================================================

class TestMonitorIndexingProgress:
    """Tests for monitor_indexing_progress function."""

    def test_monitor_with_missing_log_file(self, temp_dir):
        """Test monitor_indexing_progress with missing log file."""
        log_file = temp_dir / "nonexistent.log"

        result = monitor_indexing_progress(log_file)

        assert "error" in result
        assert "not found" in result["error"].lower()

    def test_monitor_with_empty_log_file(self, temp_dir):
        """Test monitor_indexing_progress with empty log file."""
        log_file = temp_dir / "empty.log"
        log_file.touch()

        result = monitor_indexing_progress(log_file)

        assert result["success_calls"] == 0
        assert result["errors"] == 0

    def test_monitor_counts_success_calls(self, temp_dir):
        """Test that monitor_indexing_progress counts successful API calls."""
        log_file = temp_dir / "test.log"
        log_content = """
2024-01-01 10:00:00,000 - INFO - Request sent
2024-01-01 10:00:01,000 - INFO - 200 OK - Success
2024-01-01 10:00:02,000 - INFO - 200 OK - Success
2024-01-01 10:00:03,000 - INFO - 200 OK - Success
        """
        log_file.write_text(log_content.strip())

        result = monitor_indexing_progress(log_file)

        assert result["success_calls"] == 3

    def test_monitor_counts_errors(self, temp_dir):
        """Test that monitor_indexing_progress counts error lines."""
        log_file = temp_dir / "test.log"
        log_content = """
2024-01-01 10:00:00,000 - INFO - 200 OK - Success
2024-01-01 10:00:01,000 - ERROR - Connection failed
2024-01-01 10:00:02,000 - ERROR - Timeout occurred
        """
        log_file.write_text(log_content.strip())

        result = monitor_indexing_progress(log_file)

        assert result["errors"] == 2

    def test_monitor_calculates_activity_status(self, temp_dir):
        """Test that monitor_indexing_progress determines activity status."""
        log_file = temp_dir / "test.log"

        # Create log with recent timestamp
        now = datetime.now()
        log_content = f"{now.strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]} - INFO - 200 OK"
        log_file.write_text(log_content)

        result = monitor_indexing_progress(log_file)

        assert "is_active" in result

    def test_monitor_calculates_elapsed_time(self, temp_dir):
        """Test that monitor_indexing_progress calculates elapsed time."""
        log_file = temp_dir / "test.log"
        log_content = """
2024-01-01 10:00:00,000 - INFO - 200 OK
2024-01-01 12:00:00,000 - INFO - 200 OK
        """
        log_file.write_text(log_content.strip())

        result = monitor_indexing_progress(log_file)

        assert "elapsed_hours" in result
        assert result["elapsed_hours"] == pytest.approx(2.0, abs=0.1)

    def test_monitor_calculates_processing_rate(self, temp_dir):
        """Test that monitor_indexing_progress calculates rate."""
        log_file = temp_dir / "test.log"
        log_content = """
2024-01-01 10:00:00,000 - INFO - 200 OK
2024-01-01 10:30:00,000 - INFO - 200 OK
2024-01-01 11:00:00,000 - INFO - 200 OK
        """
        log_file.write_text(log_content.strip())

        result = monitor_indexing_progress(log_file)

        assert "rate_per_hour" in result
        assert result["rate_per_hour"] > 0

    def test_monitor_estimates_completion(self, temp_dir):
        """Test that monitor_indexing_progress estimates completion time."""
        log_file = temp_dir / "test.log"
        log_content = """
2024-01-01 10:00:00,000 - INFO - 200 OK
2024-01-01 11:00:00,000 - INFO - 200 OK
        """
        log_file.write_text(log_content.strip())

        result = monitor_indexing_progress(log_file)

        assert "estimated_hours_left" in result
        assert "estimated_completion" in result

    def test_monitor_handles_malformed_timestamps(self, temp_dir):
        """Test that monitor_indexing_progress handles malformed log lines."""
        log_file = temp_dir / "test.log"
        log_content = """
Invalid log line
2024-01-01 10:00:00,000 - INFO - 200 OK
Another invalid line
        """
        log_file.write_text(log_content.strip())

        result = monitor_indexing_progress(log_file)

        # Should handle gracefully and count what it can
        assert result["success_calls"] == 1


# ============================================================================
# Integration Tests
# ============================================================================

class TestStatisticsIntegration:
    """Integration tests for statistics module."""

    def test_full_workflow_with_mock_data(self, temp_dir):
        """Test complete workflow from files to statistics."""
        # Create conversation structure
        conv_dir = temp_dir / "conversation_1"
        conv_dir.mkdir()
        (conv_dir / "Conversation.txt").write_text("Test email")

        # Create embeddings directory with chunks
        emb_dir = temp_dir / "_index" / "embeddings"
        emb_dir.mkdir(parents=True)

        chunks_data = {"chunks": [{"text": "chunk 1"}, {"text": "chunk 2"}]}
        pkl_file = emb_dir / "worker_0_batch_00000.pkl"
        with open(pkl_file, "wb") as f:
            pickle.dump(chunks_data, f)

        # Get file statistics
        file_stats = get_file_statistics(temp_dir)
        assert file_stats["total_files"] >= 1

        # Count chunks
        chunk_count = count_total_chunks(str(temp_dir))
        assert chunk_count == 2

    def test_statistics_with_multiple_conversations(self, temp_dir):
        """Test statistics with multiple conversation directories."""
        # Create multiple conversations
        for i in range(5):
            conv_dir = temp_dir / f"conversation_{i}"
            conv_dir.mkdir()
            (conv_dir / "Conversation.txt").write_text(f"Email {i}")

            # Add some attachments
            attach_dir = conv_dir / "Attachments"
            attach_dir.mkdir()
            (attach_dir / f"doc{i}.pdf").touch()

        file_stats = get_file_statistics(temp_dir)

        assert file_stats["conversation_folders"] == 5
        assert file_stats["conversation_txt_files"] == 5
        assert file_stats["total_files"] == 10  # 5 txt + 5 pdf


# ============================================================================
# Edge Case Tests
# ============================================================================

class TestStatisticsEdgeCases:
    """Edge case tests for statistics module."""

    def test_get_file_statistics_with_special_characters_in_names(self, temp_dir):
        """Test get_file_statistics with special characters in filenames."""
        conv_dir = temp_dir / "conversation_1"
        conv_dir.mkdir()

        # Create files with special characters
        (conv_dir / "file@#$%.txt").touch()
        (conv_dir / "document (copy).pdf").touch()
        (conv_dir / "data[1].json").touch()

        result = get_file_statistics(temp_dir)

        assert result["total_files"] == 3

    def test_count_total_chunks_with_empty_chunks_list(self, temp_dir):
        """Test count_total_chunks with empty chunks list."""
        emb_dir = temp_dir / "_index" / "embeddings"
        emb_dir.mkdir(parents=True)

        # Create pickle with empty chunks
        empty_data = {"chunks": []}
        pkl_file = emb_dir / "worker_0_batch_00000.pkl"
        with open(pkl_file, "wb") as f:
            pickle.dump(empty_data, f)

        result = count_total_chunks(str(temp_dir))

        assert result == 0

    def test_monitor_with_very_large_log_file(self, temp_dir):
        """Test monitor_indexing_progress with large log file."""
        log_file = temp_dir / "large.log"

        # Create log with many entries
        log_lines = []
        for i in range(1000):
            timestamp = f"2024-01-01 10:{i // 60:02d}:{i % 60:02d},000"
            log_lines.append(f"{timestamp} - INFO - 200 OK")

        log_file.write_text("\n".join(log_lines))

        result = monitor_indexing_progress(log_file)

        assert result["success_calls"] == 1000

    def test_get_file_statistics_with_symlinks(self, temp_dir):
        """Test get_file_statistics handles symlinks correctly."""
        conv_dir = temp_dir / "conversation_1"
        conv_dir.mkdir()

        # Create a real file
        real_file = conv_dir / "Conversation.txt"
        real_file.touch()

        # Note: Symlink creation might fail on Windows without admin rights
        # This test will skip gracefully if symlinks aren't supported
        try:
            link_file = conv_dir / "link.txt"
            link_file.symlink_to(real_file)
        except (OSError, NotImplementedError):
            pytest.skip("Symlinks not supported on this system")

        result = get_file_statistics(temp_dir)

        # Should count both real file and symlink
        assert result["total_files"] >= 1

    def test_count_total_chunks_with_non_pickle_files(self, temp_dir):
        """Test count_total_chunks ignores non-pickle files."""
        emb_dir = temp_dir / "_index" / "embeddings"
        emb_dir.mkdir(parents=True)

        # Create valid pickle
        valid_data = {"chunks": [{"text": "chunk"}]}
        pkl_file = emb_dir / "worker_0_batch_00000.pkl"
        with open(pkl_file, "wb") as f:
            pickle.dump(valid_data, f)

        # Create non-pickle files (should be ignored)
        (emb_dir / "readme.txt").touch()
        (emb_dir / "data.json").touch()

        result = count_total_chunks(str(temp_dir))

        # Should only count the pickle file
        assert result == 1
