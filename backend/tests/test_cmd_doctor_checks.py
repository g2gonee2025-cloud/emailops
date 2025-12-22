from unittest.mock import MagicMock, patch

import pytest
from cortex_cli.cmd_doctor import (
    _collect_failures_and_warnings,
    _probe_embeddings,
    check_db,
    check_exports,
    check_index_health,
    check_ingest,
    check_postgres,
    check_redis,
    check_reranker,
)


class TestDoctorChecks:
    @pytest.fixture
    def mock_config(self):
        conf = MagicMock()
        conf.database.url = "postgresql://user:pass@localhost:5432/db"
        conf.directories.export_root = "exports"
        conf.directories.index_dirname = "_index"
        conf.embedding.output_dimensionality = 768
        conf.search.reranker_endpoint = "http://rerank"
        return conf

    def test_check_postgres_success(self, mock_config):
        with patch("sqlalchemy.create_engine") as mock_engine:
            mock_conn = MagicMock()
            mock_engine.return_value.connect.return_value.__enter__.return_value = (
                mock_conn
            )

            success, error = check_postgres(mock_config)
            assert success is True
            assert error is None
            mock_conn.execute.assert_called()

    def test_check_postgres_fail(self, mock_config):
        with patch("sqlalchemy.create_engine") as mock_engine:
            mock_engine.side_effect = Exception("Connection fail")

            success, error = check_postgres(mock_config)
            assert success is False
            assert "Connection fail" in error

    def test_check_redis_success(self, mock_config):
        with patch("redis.from_url") as mock_redis:
            mock_client = MagicMock()
            mock_redis.return_value = mock_client

            success, error = check_redis(mock_config)
            assert success is True
            assert error is None
            mock_client.ping.assert_called()

    def test_check_redis_fail(self, mock_config):
        with patch("redis.from_url", side_effect=Exception("Redis down")):
            success, error = check_redis(mock_config)
            assert success is False
            assert "Redis down" in error

    def test_check_reranker_success(self, mock_config):
        with patch("httpx.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_get.return_value = mock_resp

            success, error = check_reranker(mock_config)
            assert success is True
            assert error is None
            mock_get.assert_called_with("http://rerank/health", timeout=5.0)

    def test_check_reranker_fail_status(self, mock_config):
        with patch("httpx.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.status_code = 500
            mock_get.return_value = mock_resp

            success, error = check_reranker(mock_config)
            assert success is False
            assert "status 500" in error

    def test_check_reranker_no_endpoint(self, mock_config):
        mock_config.search.reranker_endpoint = None
        success, error = check_reranker(mock_config)
        assert success is False
        assert "No reranker endpoint" in error

    def test_check_exports_success(self, mock_config, tmp_path):
        export_root = tmp_path / "exports"
        export_root.mkdir()

        # Create a B1 folder
        b1 = export_root / "2023-10-01"
        b1.mkdir()
        (b1 / "manifest.json").touch()

        success, folders, error = check_exports(mock_config, tmp_path)
        assert success is True
        assert "2023-10-01" in folders
        assert error is None

    def test_check_exports_missing_root(self, mock_config, tmp_path):
        success, folders, error = check_exports(mock_config, tmp_path)
        assert success is False
        assert "does not exist" in error

    def test_check_db_migrations(self, mock_config):
        with patch("sqlalchemy.create_engine") as mock_engine:
            mock_conn = MagicMock()
            mock_engine.return_value.connect.return_value.__enter__.return_value = (
                mock_conn
            )

            # Mock migration query
            mock_result = MagicMock()
            mock_result.fetchone.return_value = ("1234567890",)
            mock_conn.execute.side_effect = [
                None,
                mock_result,
            ]  # First select 1, then alembic

            success, status, error = check_db(mock_config)
            assert success is True
            assert status["connected"] is True
            assert status["migrations_current"] is True
            assert status["latest_migration"] == "1234567890"

    def test_check_index_health_success(self, mock_config, tmp_path):
        index_dir = tmp_path / "_index"
        index_dir.mkdir()
        (index_dir / "index.faiss").touch()

        with patch("cortex_cli.cmd_doctor.create_engine") as mock_engine:
            mock_conn = MagicMock()
            mock_engine.return_value.connect.return_value.__enter__.return_value = (
                mock_conn
            )

            mock_result = MagicMock()
            mock_result.dim = 768
            mock_conn.execute.return_value.fetchone.return_value = mock_result

            success, status, error = check_index_health(mock_config, tmp_path)
            if not success:
                print(f"Error: {error}")
            assert success is True
            assert status["file_count"] == 1
            assert status["db_embedding_dim"] == 768

    @patch("cortex.llm.client.embed_texts")
    def test_probe_embeddings_success(self, mock_embed):
        mock_embed.return_value = [[0.1, 0.2, 0.3]]
        success, dim = _probe_embeddings("vertex")
        assert success is True
        assert dim == 3

    @patch("cortex.llm.client.embed_texts")
    def test_probe_embeddings_fail(self, mock_embed):
        mock_embed.side_effect = Exception("API error")
        success, dim = _probe_embeddings("vertex")
        assert success is False
        assert dim is None

    def test_check_ingest_no_sample(self, mock_config, tmp_path):
        with patch("cortex_cli.cmd_doctor._find_sample_file", return_value=None):
            success, details, error = check_ingest(mock_config, tmp_path)
            assert success is True  # Returns True if no sample found, but with info??
            # Logic: return True, details, "No sample messages found..."
            assert "No sample messages found" in error
            assert details["sample_found"] is False

    def test_collect_failures(self):
        f, w = _collect_failures_and_warnings(
            dep_error=True,
            index_error=False,
            embed_error=True,
            db_error=False,
            redis_error=False,
            rerank_error=False,
            exp_warning=True,
            ing_warning=False,
            missing_optional=True,
        )
        assert "Missing critical dependencies" in f
        assert "Embedding connectivity failed" in f
        assert "Export root issues" in w
        assert "Missing optional packages" in w
