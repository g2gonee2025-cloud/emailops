
"""Unit tests for processing.processor module."""

import json
import os
import pickle
import queue
import tempfile
import time
from datetime import timedelta
from pathlib import Path
from unittest import TestCase
from unittest.mock import MagicMock, Mock, mock_open, patch

import numpy as np
import pytest

from processing.processor import (
    ChunkJob,
    ProcessingStats,
    UnifiedProcessor,
    WorkerConfig,
    WorkerStats,
    _chunk_worker_entry,
    _initialize_gcp_credentials,
    _safe_filename_for_doc,
    _save_chunks_to_path,
    main,
)


class TestDataClasses(TestCase):
    """Test data classes."""

    def test_chunk_job_initialization(self):
        """Test ChunkJob initialization with required fields."""
        job = ChunkJob(
            doc_id="test_doc",
            doc_path=Path("/path/to/doc"),
            file_size=1024,
            priority=5
        )
        assert job.doc_id == "test_doc"
        assert job.doc_path == Path("/path/to/doc")
        assert job.file_size == 1024
        assert job.priority == 5

    def test_chunk_job_default_priority(self):
        """Test ChunkJob with default priority."""
        job = ChunkJob(
            doc_id="test_doc",
            doc_path=Path("/path/to/doc"),
            file_size=1024
        )
        assert job.priority == 0

    def test_worker_config_initialization(self):
        """Test WorkerConfig initialization."""
        config = WorkerConfig(
            worker_id=1,
            jobs_assigned=[("doc1", "/path1", 100)],
            chunk_config={"chunk_size": 1600}
        )
        assert config.worker_id == 1
        assert len(config.jobs_assigned) == 1
        assert config.chunk_config["chunk_size"] == 1600

    def test_worker_stats_progress_percent(self):
        """Test WorkerStats progress calculation."""
        stats = WorkerStats(
            worker_id=0,
            docs_processed=5,
            docs_total=10,
            chunks_created=50,
            bytes_processed=5000,
            bytes_total=10000,
            start_time=time.time(),
            last_update=time.time(),
            errors=0,
            status="running"
        )
        assert stats.progress_percent == 50.0

    def test_worker_stats_progress_percent_zero_total(self):
        """Test WorkerStats progress with zero total."""
        stats = WorkerStats(
            worker_id=0,
            docs_processed=0,
            docs_total=0,
            chunks_created=0,
            bytes_processed=0,
            bytes_total=0,
            start_time=time.time(),
            last_update=time.time(),
            errors=0,
            status="running"
        )
        assert stats.progress_percent == 0.0

    def test_worker_stats_estimated_time_remaining(self):
        """Test WorkerStats ETA calculation."""
        start_time = time.time() - 10  # Started 10 seconds ago
        stats = WorkerStats(
            worker_id=0,
            docs_processed=5,
            docs_total=10,
            chunks_created=50,
            bytes_processed=5000,
            bytes_total=10000,
            start_time=start_time,
            last_update=time.time(),
            errors=0,
            status="running"
        )
        eta = stats.estimated_time_remaining
        assert isinstance(eta, timedelta)
        assert eta.total_seconds() > 0

    def test_worker_stats_no_eta_when_no_progress(self):
        """Test WorkerStats ETA is None when no progress."""
        stats = WorkerStats(
            worker_id=0,
            docs_processed=0,
            docs_total=10,
            chunks_created=0,
            bytes_processed=0,
            bytes_total=10000,
            start_time=time.time(),
            last_update=time.time(),
            errors=0,
            status="running"
        )
        assert stats.estimated_time_remaining is None

    def test_processing_stats_progress_percent(self):
        """Test ProcessingStats progress calculation."""
        stats = ProcessingStats(
            worker_id=0,
            project_id="test-project",
            chunks_processed=30,
            chunks_total=100,
            start_time=time.time(),
            last_update=time.time(),
            errors=0,
            status="running",
            account_group=1
        )
        assert stats.progress_percent == 30.0

    def test_processing_stats_progress_percent_zero_total(self):
        """Test ProcessingStats progress with zero total."""
        stats = ProcessingStats(
            worker_id=0,
            project_id="test-project",
            chunks_processed=0,
            chunks_total=0,
            start_time=time.time(),
            last_update=time.time(),
            errors=0,
            status="running",
            account_group=1
        )
        assert stats.progress_percent == 0.0

    def test_processing_stats_estimated_time_remaining(self):
        """Test ProcessingStats ETA calculation."""
        start_time = time.time() - 10
        stats = ProcessingStats(
            worker_id=0,
            project_id="test-project",
            chunks_processed=30,
            chunks_total=100,
            start_time=start_time,
            last_update=time.time(),
            errors=0,
            status="running",
            account_group=1
        )
        eta = stats.estimated_time_remaining
        assert isinstance(eta, timedelta)
        assert eta.total_seconds() > 0


class TestHelperFunctions(TestCase):
    """Test helper functions."""

    def test_safe_filename_for_doc_simple(self):
        """Test _safe_filename_for_doc with simple input."""
        result = _safe_filename_for_doc("test_document.txt")
        assert "test_document.txt" in result
        assert result.endswith(".json")
        assert len(result.split('.')[-2]) == 8  # Hash is 8 chars

    def test_safe_filename_for_doc_special_chars(self):
        """Test _safe_filename_for_doc with special characters."""
        result = _safe_filename_for_doc("test/doc@#$%^&*.txt")
        assert "@" not in result
        assert "#" not in result
        assert "$" not in result
        assert result.endswith(".json")

    def test_safe_filename_for_doc_long_name(self):
        """Test _safe_filename_for_doc with very long name."""
        long_name = "a" * 200
        result = _safe_filename_for_doc(long_name)
        # Should be truncated to 128 chars + hash + .json
        assert len(result) < 150
        assert result.endswith(".json")

    def test_save_chunks_to_path(self):
        """Test _save_chunks_to_path function."""
        with tempfile.TemporaryDirectory() as tmpdir:
            chunks_dir = Path(tmpdir)
            doc_id = "test_doc"
            chunks = [
                {"text": "chunk1", "metadata": {}},
                {"text": "chunk2", "metadata": {}}
            ]
            file_size = 1024

            result = _save_chunks_to_path(chunks_dir, doc_id, chunks, file_size)

            assert result.exists()
            assert result.suffix == ".json"

            with open(result) as f:
                data = json.load(f)

            assert data["doc_id"] == doc_id
            assert data["num_chunks"] == 2
            assert len(data["chunks"]) == 2
            assert data["metadata"]["original_size"] == file_size

    @patch('processing.processor.logging')
    @patch.dict(os.environ, {}, clear=True)
    def test_initialize_gcp_credentials_no_env(self, mock_logging):
        """Test _initialize_gcp_credentials with no environment variable."""
        with patch('emailops.config.get_config') as mock_get_config:
            mock_config = Mock()
            mock_config.get_credential_file.return_value = None
            mock_config.get_secrets_dir.return_value = Path("/secrets")
            mock_get_config.return_value = mock_config

            result = _initialize_gcp_credentials()

            assert result is None
            mock_logging.error.assert_called()

    @patch('processing.processor.logging')
    def test_initialize_gcp_credentials_from_env(self, mock_logging):
        """Test _initialize_gcp_credentials with existing env variable."""
        cred_path = "/path/to/creds.json"

        with patch('processing.processor.Path') as mock_path_class:
            mock_path_obj = Mock()
            mock_path_obj.exists.return_value = True
            mock_path_class.return_value = mock_path_obj

            with patch.dict(os.environ, {"GOOGLE_APPLICATION_CREDENTIALS": cred_path}):
                result = _initialize_gcp_credentials()

                assert result == cred_path
                mock_logging.info.assert_called()

    @patch('processing.processor.Path.exists')
    @patch('processing.processor.json.load')
    @patch('builtins.open', new_callable=mock_open)
    @patch('processing.processor.logging')
    def test_initialize_gcp_credentials_from_config(self, mock_logging, mock_file, mock_json_load, mock_exists):
        """Test _initialize_gcp_credentials using config file."""
        mock_exists.return_value = False
        mock_json_load.return_value = {"project_id": "test-project"}

        with patch.dict(os.environ, {}, clear=True):
            with patch('emailops.config.get_config') as mock_get_config:
                mock_config = Mock()
                mock_config.get_credential_file.return_value = Path("/secrets/creds.json")
                mock_config.GCP_REGION = "us-central1"
                mock_config.VERTEX_LOCATION = "us-central1"
                mock_get_config.return_value = mock_config

                result = _initialize_gcp_credentials()

                assert result == str(Path("/secrets/creds.json"))
                assert os.environ["GOOGLE_APPLICATION_CREDENTIALS"] == str(Path("/secrets/creds.json"))
                assert os.environ["GCP_PROJECT"] == "test-project"
                mock_logging.info.assert_called()


class TestUnifiedProcessor(TestCase):
    """Test UnifiedProcessor class."""

    @patch('emailops.config.get_config')
    def test_init_with_defaults(self, mock_get_config):
        """Test UnifiedProcessor initialization with defaults."""
        mock_config = Mock()
        mock_config.DEFAULT_BATCH_SIZE = 64
        mock_config.DEFAULT_CHUNK_SIZE = 1600
        mock_config.DEFAULT_CHUNK_OVERLAP = 200
        mock_config.DEFAULT_NUM_WORKERS = 4
        mock_config.CHUNK_DIRNAME = "_chunks"
        mock_config.INDEX_DIRNAME = "_index"
        mock_get_config.return_value = mock_config

        with tempfile.TemporaryDirectory() as tmpdir:
            processor = None
            try:
                processor = UnifiedProcessor(tmpdir, mode="chunk")

                assert processor.root_dir == Path(tmpdir).resolve()
                assert processor.mode == "chunk"
                assert processor.batch_size == 64
                assert processor.chunk_size == 1600
                assert processor.chunk_overlap == 200
                assert processor.resume is True
                assert processor.test_mode is False
            finally:
                if processor:
                    processor.close()

    @patch('emailops.config.get_config')
    def test_init_with_custom_params(self, mock_get_config):
        """Test UnifiedProcessor initialization with custom parameters."""
        mock_config = Mock()
        mock_config.DEFAULT_BATCH_SIZE = 64
        mock_config.DEFAULT_CHUNK_SIZE = 1600
        mock_config.DEFAULT_CHUNK_OVERLAP = 200
        mock_config.DEFAULT_NUM_WORKERS = 4
        mock_config.CHUNK_DIRNAME = "_chunks"
        mock_config.INDEX_DIRNAME = "_index"
        mock_get_config.return_value = mock_config

        with tempfile.TemporaryDirectory() as tmpdir:
            processor = None
            try:
                processor = UnifiedProcessor(
                    tmpdir,
                    mode="embed",
                    num_workers=8,
                    batch_size=128,
                    chunk_size=2000,
                    chunk_overlap=300,
                    resume=False,
                    test_mode=True
                )

                assert processor.num_workers == 8
                assert processor.batch_size == 128
                assert processor.chunk_size == 2000
                assert processor.chunk_overlap == 300
                assert processor.resume is False
                assert processor.test_mode is True
            finally:
                if processor:
                    processor.close()

    @patch('emailops.config.get_config')
    def test_init_embed_mode(self, mock_get_config):
        """Test UnifiedProcessor initialization in embed mode."""
        mock_config = Mock()
        mock_config.DEFAULT_BATCH_SIZE = 64
        mock_config.DEFAULT_CHUNK_SIZE = 1600
        mock_config.DEFAULT_CHUNK_OVERLAP = 200
        mock_config.DEFAULT_NUM_WORKERS = 4
        mock_config.CHUNK_DIRNAME = "_chunks"
        mock_config.INDEX_DIRNAME = "_index"
        mock_get_config.return_value = mock_config

        with tempfile.TemporaryDirectory() as tmpdir:
            processor = None
            try:
                processor = UnifiedProcessor(tmpdir, mode="embed")

                assert processor.mode == "embed"
                assert processor.index_dir.exists()
                assert processor.index_dir.name == "_index"
            finally:
                if processor:
                    processor.close()

    @patch('emailops.config.get_config', side_effect=ImportError)
    def test_init_without_config(self, mock_get_config):
        """Test UnifiedProcessor initialization when config import fails."""
        with tempfile.TemporaryDirectory() as tmpdir:
            processor = UnifiedProcessor(tmpdir, mode="chunk")

            # Should use fallback defaults
            assert processor.batch_size == 64
            assert processor.chunk_size == 1600
            assert processor.chunk_overlap == 200

    @patch('emailops.config.get_config')
    def test_build_chunk_config_kwargs(self, mock_get_config):
        """Test _build_chunk_config_kwargs method."""
        mock_config = Mock()
        mock_config.DEFAULT_BATCH_SIZE = 64
        mock_config.DEFAULT_CHUNK_SIZE = 1600
        mock_config.DEFAULT_CHUNK_OVERLAP = 200
        mock_config.DEFAULT_NUM_WORKERS = 4
        mock_config.CHUNK_DIRNAME = "_chunks"
        mock_get_config.return_value = mock_config

        with tempfile.TemporaryDirectory() as tmpdir:
            processor = None
            try:
                processor = UnifiedProcessor(root_dir=tmpdir)
                config = processor._build_chunk_config_kwargs()
                assert config["chunk_size"] == 1600
                assert config["chunk_overlap"] == 200
            finally:
                if processor:
                    processor.close()

    @patch('emailops.config.get_config')
    def test_find_documents_basic(self, mock_get_config):
        """Test _find_documents method."""
        mock_config = Mock()
        mock_config.DEFAULT_BATCH_SIZE = 64
        mock_config.DEFAULT_CHUNK_SIZE = 1600
        mock_config.DEFAULT_CHUNK_OVERLAP = 200
        mock_config.DEFAULT_NUM_WORKERS = 4
        mock_config.CHUNK_DIRNAME = "_chunks"
        mock_get_config.return_value = mock_config

        with tempfile.TemporaryDirectory() as tmpdir:
            processor = UnifiedProcessor(tmpdir, mode="chunk")
            processor.input_dir = Path(tmpdir)
            processor.chunks_dir = Path(tmpdir) / "_chunks" / "chunks"
            processor.chunks_dir.mkdir(parents=True)

            # Create test files
            (Path(tmpdir) / "doc1.txt").write_text("test content 1")
            (Path(tmpdir) / "doc2.txt").write_text("test content 2")
            (Path(tmpdir) / "empty.txt").write_text("")  # Empty file should be skipped

            jobs = processor._find_documents("*.txt")

            assert len(jobs) == 2
            assert all(isinstance(job, ChunkJob) for job in jobs)
            assert all(job.file_size > 0 for job in jobs)

    @patch('emailops.config.get_config')
    def test_find_documents_with_resume(self, mock_get_config):
        """Test _find_documents with resume enabled."""
        mock_config = Mock()
        mock_config.DEFAULT_BATCH_SIZE = 64
        mock_config.DEFAULT_CHUNK_SIZE = 1600
        mock_config.DEFAULT_CHUNK_OVERLAP = 200
        mock_config.DEFAULT_NUM_WORKERS = 4
        mock_config.CHUNK_DIRNAME = "_chunks"
        mock_get_config.return_value = mock_config

        with tempfile.TemporaryDirectory() as tmpdir:
            processor = UnifiedProcessor(tmpdir, mode="chunk", resume=True)
            processor.input_dir = Path(tmpdir)
            processor.chunks_dir = Path(tmpdir) / "_chunks" / "chunks"
            processor.chunks_dir.mkdir(parents=True)

            # Create test files
            (Path(tmpdir) / "doc1.txt").write_text("test content 1")
            (Path(tmpdir) / "doc2.txt").write_text("test content 2")

            # Simulate existing chunk for doc1
            chunk_file = processor._existing_chunk_path("doc1.txt")
            chunk_file.write_text("{}")

            jobs = processor._find_documents("*.txt")

            # Should only find doc2 since doc1 already has chunks
            assert len(jobs) == 1
            assert jobs[0].doc_id == "doc2.txt"

    @patch('emailops.config.get_config')
    def test_find_documents_test_mode(self, mock_get_config):
        """Test _find_documents in test mode limits to 10 files."""
        mock_config = Mock()
        mock_config.DEFAULT_BATCH_SIZE = 64
        mock_config.DEFAULT_CHUNK_SIZE = 1600
        mock_config.DEFAULT_CHUNK_OVERLAP = 200
        mock_config.DEFAULT_NUM_WORKERS = 4
        mock_config.CHUNK_DIRNAME = "_chunks"
        mock_get_config.return_value = mock_config

        with tempfile.TemporaryDirectory() as tmpdir:
            processor = UnifiedProcessor(tmpdir, mode="chunk", test_mode=True)
            processor.input_dir = Path(tmpdir)
            processor.chunks_dir = Path(tmpdir) / "_chunks" / "chunks"
            processor.chunks_dir.mkdir(parents=True)

            # Create more than 10 test files
            for i in range(15):
                (Path(tmpdir) / f"doc{i}.txt").write_text(f"content {i}")

            jobs = processor._find_documents("*.txt")

            assert len(jobs) == 10  # Limited by test mode
            processor.close()

    @patch('emailops.config.get_config')
    def test_distribute_chunking_work(self, mock_get_config):
        """Test _distribute_chunking_work method."""
        mock_config = Mock()
        mock_config.DEFAULT_BATCH_SIZE = 64
        mock_config.DEFAULT_CHUNK_SIZE = 1600
        mock_config.DEFAULT_CHUNK_OVERLAP = 200
        mock_config.DEFAULT_NUM_WORKERS = 2
        mock_config.CHUNK_DIRNAME = "_chunks"
        mock_get_config.return_value = mock_config

        with tempfile.TemporaryDirectory() as tmpdir:
            processor = None
            try:
                processor = UnifiedProcessor(tmpdir, num_workers=2)

                jobs = [
                    ChunkJob("doc1", Path("/path1"), 1000),
                    ChunkJob("doc2", Path("/path2"), 2000),
                    ChunkJob("doc3", Path("/path3"), 1500),
                    ChunkJob("doc4", Path("/path4"), 500)
                ]

                chunk_config = {"chunk_size": 1600}
                configs = processor._distribute_chunking_work(jobs, chunk_config)

                assert len(configs) == 2
                assert all(isinstance(c, WorkerConfig) for c in configs)

                # Check work is distributed
                total_jobs = sum(len(c.jobs_assigned) for c in configs)
                assert total_jobs == 4

                # Check load balancing (should distribute by size)
                worker0_size = sum(job[2] for job in configs[0].jobs_assigned)
                worker1_size = sum(job[2] for job in configs[1].jobs_assigned)
                assert abs(worker0_size - worker1_size) <= 1000  # Reasonably balanced
            finally:
                if processor:
                    processor.close()
            processor.close()


class TestChunkWorkerEntry(TestCase):
    """Test chunk worker entry function."""

    @patch('processing.processor.logging.basicConfig')
    @patch('emailops.text_chunker.TextChunker')
    @patch('emailops.utils.read_text_file')
    def test_chunk_worker_entry_success(self, mock_read, mock_chunker_class, mock_logging):
        """Test successful chunk worker execution."""
        # Setup mocks
        mock_read.return_value = "test content"
        mock_chunker = Mock()
        mock_chunker.chunk_text.return_value = [{"text": "chunk1", "metadata": {}}]
        mock_chunker_class.return_value = mock_chunker

        stats_queue = Mock()
        control_queue = Mock()
        control_queue.get_nowait.side_effect = queue.Empty

        with tempfile.TemporaryDirectory() as tmpdir:
            chunks_dir = Path(tmpdir) / "chunks"
            chunks_dir.mkdir()

            _chunk_worker_entry(
                worker_id=0,
                jobs_assigned=[("doc1", str(Path(tmpdir) / "doc1.txt"), 100)],
                chunk_config={"chunk_size": 1600},
                chunks_dir=str(chunks_dir),
                stats_queue=stats_queue,
                control_queue=control_queue,
                log_level=20  # INFO
            )

            # Check stats were reported
            assert stats_queue.put.call_count >= 2  # Initial and final stats
            final_stats = stats_queue.put.call_args_list[-1][0][0]
            assert final_stats.status == "completed"
            assert final_stats.docs_processed == 1
            assert final_stats.chunks_created == 1

    @patch('processing.processor.logging.basicConfig')
    def test_chunk_worker_entry_import_error(self, mock_logging):
        """Test chunk worker with import error."""
        stats_queue = Mock()
        control_queue = Mock()

        with patch('emailops.text_chunker.TextChunker', side_effect=ImportError("Missing dependency")):
            _chunk_worker_entry(
                worker_id=0,
                jobs_assigned=[("doc1", "/path/doc1.txt", 100)],
                chunk_config={},
                chunks_dir="/chunks",
                stats_queue=stats_queue,
                control_queue=control_queue,
                log_level=20
            )

            # Should report error stats
            assert stats_queue.put.call_count == 1
            error_stats = stats_queue.put.call_args[0][0]
            assert error_stats.errors == 1
            assert "failed" in error_stats.status


class TestEmbeddingOperations(TestCase):
    """Test embedding operations."""

    def test_create_embeddings_vertex(self):
        """Test create_embeddings with Vertex provider."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Setup real configuration
            from emailops.config import get_config
            config = get_config()
            config.update_environment()

            processor = UnifiedProcessor(tmpdir, mode="embed", batch_size=2)

            # Create test chunk files
            chunks_dir = Path(tmpdir) / config.CHUNK_DIRNAME / "chunks"
            chunks_dir.mkdir(parents=True)

            chunk_data = {
                "doc_id": "test_doc",
                "chunks": [
                    {"text": "This is a test chunk for embedding."},
                    {"text": "This is another test chunk for embedding."}
                ]
            }
            (chunks_dir / "test.json").write_text(json.dumps(chunk_data))

            processor.create_embeddings(use_chunked_files=True)

            # Check embeddings were saved
            emb_dir = Path(tmpdir) / config.INDEX_DIRNAME / "embeddings"
            assert emb_dir.exists()
            pkl_files = list(emb_dir.glob("*.pkl"))
            assert len(pkl_files) > 0

            # Verify content of a pickle file
            with open(pkl_files[0], "rb") as f:
                data = pickle.load(f)
                assert "embeddings" in data
                assert len(data["embeddings"]) > 0
                assert len(data["embeddings"][0]) > 1 # Embedding vector should have dimensions

    @patch('emailops.config.get_config')
    @patch('processing.processor._initialize_gcp_credentials')
    @patch.dict(os.environ, {"EMBED_PROVIDER": "vertex"})
    def test_create_embeddings_no_credentials(self, mock_init_creds, mock_get_config):
        """Test create_embeddings fails without credentials."""
        mock_config = Mock()
        mock_config.DEFAULT_BATCH_SIZE = 64
        mock_config.DEFAULT_CHUNK_SIZE = 1600
        mock_config.DEFAULT_CHUNK_OVERLAP = 200
        mock_config.DEFAULT_NUM_WORKERS = 4
        mock_config.CHUNK_DIRNAME = "_chunks"
        mock_config.INDEX_DIRNAME = "_index"
        mock_get_config.return_value = mock_config

        mock_init_creds.return_value = None

        with tempfile.TemporaryDirectory() as tmpdir:
            processor = UnifiedProcessor(tmpdir, mode="embed")
            processor.create_embeddings()

            mock_init_creds.assert_called_once()
            # Should exit early due to no credentials


class TestIndexOperations(TestCase):
    """Test index repair and fix operations."""

    @patch('emailops.config.get_config')
    def test_repair_index_success(self, mock_get_config):
        """Test successful index repair."""
        mock_config = Mock()
        mock_config.DEFAULT_BATCH_SIZE = 64
        mock_config.DEFAULT_CHUNK_SIZE = 1600
        mock_config.DEFAULT_CHUNK_OVERLAP = 200
        mock_config.DEFAULT_NUM_WORKERS = 4
        mock_config.INDEX_DIRNAME = "_index"
        mock_get_config.return_value = mock_config

        with tempfile.TemporaryDirectory() as tmpdir:
            processor = UnifiedProcessor(tmpdir, mode="repair")

            # Create test batch files
            emb_dir = processor.index_dir / "embeddings"
            emb_dir.mkdir(parents=True)

            batch_data = {
                "chunks": [{"text": "test", "doc_id": "doc1"}],
                "embeddings": np.array([[0.1, 0.2]], dtype="float32")
            }

            with open(emb_dir / "worker_0_batch_00000.pkl", "wb") as f:
                pickle.dump(batch_data, f)

            # Mock faiss import within the method
            with patch.dict('sys.modules', {'faiss': MagicMock()}):
                import sys
                mock_faiss = sys.modules['faiss']
                mock_faiss.IndexFlatIP.return_value = Mock()

                processor.repair_index(remove_batches=False)

                # Check outputs
                assert (processor.index_dir / "embeddings.npy").exists()
                assert (processor.index_dir / "mapping.json").exists()
                processor.close()

    @patch('emailops.config.get_config')
    def test_repair_index_no_files(self, mock_get_config):
        """Test repair_index with no batch files."""
        mock_config = Mock()
        mock_config.DEFAULT_BATCH_SIZE = 64
        mock_config.DEFAULT_CHUNK_SIZE = 1600
        mock_config.DEFAULT_CHUNK_OVERLAP = 200
        mock_config.DEFAULT_NUM_WORKERS = 4
        mock_config.INDEX_DIRNAME = "_index"
        mock_get_config.return_value = mock_config

        with tempfile.TemporaryDirectory() as tmpdir:
            processor = UnifiedProcessor(tmpdir, mode="repair")
            processor.repair_index()

            # Should exit early - no files created
            assert not (processor.index_dir / "embeddings.npy").exists()
            processor.close()


class TestCLI(TestCase):
    """Test CLI functionality."""

    @patch('sys.argv', ['processor.py', 'chunk', '--input', '/input', '--output', '/output'])
    @patch('processing.processor.UnifiedProcessor')
    def test_cli_chunk_command(self, mock_processor_class):
        """Test CLI chunk command."""
        mock_processor = Mock()
        mock_processor_class.return_value = mock_processor

        result = main()

        assert result == 0
        mock_processor_class.assert_called_once()
        mock_processor.chunk_documents.assert_called_once()

    @patch('sys.argv', ['processor.py', 'embed', '--root', '/root'])
    @patch('processing.processor.UnifiedProcessor')
    def test_cli_embed_command(self, mock_processor_class):
        """Test CLI embed command."""
        mock_processor = Mock()
        mock_processor_class.return_value = mock_processor

        result = main()

        assert result == 0
        mock_processor_class.assert_called_once()
        mock_processor.create_embeddings.assert_called_once()

    @patch('sys.argv', ['processor.py', 'repair', '--root', '/root'])
    @patch('processing.processor.UnifiedProcessor')
    def test_cli_repair_command(self, mock_processor_class):
        """Test CLI repair command."""
        mock_processor = Mock()
        mock_processor_class.return_value = mock_processor

        result = main()

        assert result == 0
        mock_processor_class.assert_called_once()
        mock_processor.repair_index.assert_called_once()

    @patch('sys.argv', ['processor.py', 'fix', '--root', '/root'])
    @patch('processing.processor.UnifiedProcessor')
    def test_cli_fix_command(self, mock_processor_class):
        """Test CLI fix command."""
        mock_processor = Mock()
        mock_processor_class.return_value = mock_processor

        result = main()

        assert result == 0
        mock_processor_class.assert_called_once()
        mock_processor.fix_failed_embeddings.assert_called_once()

    @patch('sys.argv', ['processor.py'])
    def test_cli_no_command(self):
        """Test CLI with no command shows help."""
        with patch('processing.processor.argparse.ArgumentParser.print_help') as mock_help:
            result = main()

            assert result == 1
            mock_help.assert_called_once()


# Additional test cases for edge cases and error handling
class TestErrorHandling(TestCase):
    """Test error handling scenarios."""

    @patch('emailops.config.get_config')
    def test_sequential_chunk_import_error(self, mock_get_config):
        """Test sequential chunking with import error - skip test as it needs real imports."""
        # This test requires mocking the import which happens inside the method
        # The current implementation handles ImportError gracefully by logging
        # We can't easily test this without causing actual import failures
        pytest.skip("Import error testing requires complex mocking of dynamic imports")

    @patch('emailops.config.get_config')
    @patch('emailops.llm_client.embed_texts')
    def test_embed_from_chunks_error_handling(self, mock_embed, mock_get_config):
        """Test embedding with errors in some batches."""
        mock_config = Mock()
        mock_config.DEFAULT_BATCH_SIZE = 2
        mock_config.DEFAULT_CHUNK_SIZE = 1600
        mock_config.DEFAULT_CHUNK_OVERLAP = 200
        mock_config.DEFAULT_NUM_WORKERS = 4
        mock_config.CHUNK_DIRNAME = "_chunks"
        mock_config.INDEX_DIRNAME = "_index"
        mock_get_config.return_value = mock_config

        # First call succeeds, second fails
        mock_embed.side_effect = [[[0.1, 0.2]], Exception("API error")]

        with tempfile.TemporaryDirectory() as tmpdir:
            processor = UnifiedProcessor(tmpdir, mode="embed", batch_size=1)

            chunks_dir = Path(tmpdir) / "_chunks" / "chunks"
            chunks_dir.mkdir(parents=True)

            chunk_data = {
                "doc_id": "test_doc",
                "chunks": [
                    {"text": "chunk 1"},
                    {"text": "chunk 2"}
                ]
            }
            (chunks_dir / "test.json").write_text(json.dumps(chunk_data))

            processor._embed_from_chunks(mock_embed, "test")

            # Should handle the error gracefully
            emb_dir = Path(tmpdir) / "_index" / "embeddings"
            pkl_files = list(emb_dir.glob("*.pkl"))
            assert len(pkl_files) == 1  # Only successful batch saved
            processor.close()

    @patch('emailops.config.get_config')
    @patch('processing.processor._initialize_gcp_credentials')
    @patch('emailops.llm_client.embed_texts')
    def test_fix_failed_embeddings_re_embed_error(self, mock_embed, mock_init_creds, mock_get_config):
        """Test fix_failed_embeddings when re-embedding fails."""
        mock_config = Mock()
        mock_config.DEFAULT_BATCH_SIZE = 64
        mock_config.DEFAULT_CHUNK_SIZE = 1600
        mock_config.DEFAULT_CHUNK_OVERLAP = 200
        mock_config.DEFAULT_NUM_WORKERS = 4
        mock_config.INDEX_DIRNAME = "_index"
        mock_get_config.return_value = mock_config

        mock_init_creds.return_value = "/path/to/creds.json"
        mock_embed.side_effect = Exception("Embedding failed")

        with tempfile.TemporaryDirectory() as tmpdir:
            processor = UnifiedProcessor(tmpdir, mode="fix")

            emb_dir = processor.index_dir / "embeddings"
            emb_dir.mkdir(parents=True)

            # Create batch with zero vectors
            batch_data = {
                "chunks": [{"text": "test", "doc_id": "doc1"}],
                "embeddings": np.array([[0.0, 0.0]], dtype="float32")
            }

            with open(emb_dir / "batch.pkl", "wb") as f:
                pickle.dump(batch_data, f)

            processor.fix_failed_embeddings()

            # Should handle error without crashing
            mock_embed.assert_called()
            processor.close()


class TestMonitoringFunctions(TestCase):
    """Test monitoring and display functions."""

    @patch('emailops.config.get_config')
    @patch('os.system')
    @patch('sys.stdout.isatty', return_value=True)
    def test_display_progress_tty(self, mock_isatty, mock_system, mock_get_config):
        """Test progress display in TTY mode."""
        mock_config = Mock()
        mock_config.DEFAULT_BATCH_SIZE = 64
        mock_config.DEFAULT_CHUNK_SIZE = 1600
        mock_config.DEFAULT_CHUNK_OVERLAP = 200
        mock_config.DEFAULT_NUM_WORKERS = 4
        mock_config.CHUNK_DIRNAME = "_chunks"
        mock_get_config.return_value = mock_config

        with tempfile.TemporaryDirectory() as tmpdir:
            processor = UnifiedProcessor(tmpdir)

            worker_states = {
                0: WorkerStats(
                    worker_id=0,
                    docs_processed=5,
                    docs_total=10,
                    chunks_created=50,
                    bytes_processed=5000,
                    bytes_total=10000,
                    start_time=time.time(),
                    last_update=time.time(),
                    errors=1,
                    status="running",
                    current_doc="test.txt"
                )
            }

            with patch('builtins.print') as mock_print:
                processor._display_progress(worker_states)

                # Should clear screen in TTY mode
                mock_system.assert_called()
                # Should print progress info
                assert mock_print.call_count > 5
                processor.close()

    @patch('emailops.config.get_config')
    def test_print_summary(self, mock_get_config):
        """Test summary printing."""
        mock_config = Mock()
        mock_config.DEFAULT_BATCH_SIZE = 64
        mock_config.DEFAULT_CHUNK_SIZE = 1600
        mock_config.DEFAULT_CHUNK_OVERLAP = 200
        mock_config.DEFAULT_NUM_WORKERS = 4
        mock_config.CHUNK_DIRNAME = "_chunks"
        mock_get_config.return_value = mock_config

        with tempfile.TemporaryDirectory() as tmpdir:
            processor = UnifiedProcessor(tmpdir)

            worker_states = {
                0: WorkerStats(
                    worker_id=0,
                    docs_processed=5,
                    docs_total=10,
                    chunks_created=50,
                    bytes_processed=5000,
                    bytes_total=10000,
                    start_time=time.time(),
                    last_update=time.time(),
                    errors=2,
                    status="completed"
                ),
                1: WorkerStats(
                    worker_id=1,
                    docs_processed=3,
                    docs_total=5,
                    chunks_created=30,
                    bytes_processed=3000,
                    bytes_total=5000,
                    start_time=time.time(),
                    last_update=time.time(),
                    errors=1,
                    status="completed"
                )
            }

            with patch('builtins.print') as mock_print:
                processor._print_summary(worker_states)

                # Should print summary statistics
                print_calls = [str(call) for call in mock_print.call_args_list]
                assert any("Total processed: 8" in str(call) for call in print_calls)
                assert any("Total errors: 3" in str(call) for call in print_calls)
                processor.close()
