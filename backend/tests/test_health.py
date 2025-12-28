from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from cortex.config.loader import EmailOpsConfig
from cortex.health import (
    check_postgres,
    check_redis,
    check_reranker,
    probe_embeddings,
)


@pytest.fixture
def mock_config():
    """Provides a mock CortexConfig object."""
    config = MagicMock(autospec=EmailOpsConfig)
    config.database.url = "postgresql://user:pass@localhost/db"
    config.redis.url = "redis://localhost:6379"
    config.core.provider = "test_provider"
    config.search.reranker_endpoint = "http://reranker:8000"
    return config


@pytest.mark.asyncio
async def test_check_postgres_success(mock_config):
    with patch("sqlalchemy.create_engine") as mock_create_engine:
        mock_conn = MagicMock()
        mock_create_engine.return_value.connect.return_value.__enter__.return_value = (
            mock_conn
        )
        result = await check_postgres(mock_config)
        assert result.name == "PostgreSQL"
        assert result.status == "pass"
        assert result.message == "Connected"


@pytest.mark.asyncio
async def test_check_postgres_failure(mock_config):
    with patch(
        "sqlalchemy.create_engine", side_effect=Exception("Connection timed out")
    ):
        result = await check_postgres(mock_config)
        assert result.name == "PostgreSQL"
        assert result.status == "fail"
        assert "Connection timed out" in result.message


@pytest.mark.asyncio
async def test_check_postgres_no_config():
    config = MagicMock(autospec=EmailOpsConfig)
    config.database = None
    result = await check_postgres(config)
    assert result.status == "fail"
    assert result.message == "Database URL not configured"


@pytest.mark.asyncio
async def test_check_redis_success(mock_config):
    with patch("redis.from_url") as mock_from_url:
        mock_redis_client = MagicMock()
        mock_from_url.return_value = mock_redis_client
        result = await check_redis(mock_config)
        assert result.name == "Redis"
        assert result.status == "pass"
        mock_redis_client.ping.assert_called_once()


@pytest.mark.asyncio
async def test_check_redis_failure(mock_config):
    with patch("redis.from_url", side_effect=Exception("Auth error")):
        result = await check_redis(mock_config)
        assert result.name == "Redis"
        assert result.status == "fail"
        assert "Auth error" in result.message


@pytest.mark.asyncio
async def test_probe_embeddings_success(mock_config):
    import numpy as np

    with patch("cortex.llm.client.embed_texts") as mock_embed:
        # Mock numpy array with shape attribute
        mock_result = np.array([[0.1] * 384])
        mock_embed.return_value = mock_result
        result = await probe_embeddings(mock_config)
        assert result.name == "Embeddings API"
        assert result.status == "pass"
        assert "Dimension: 384" in result.message
        assert result.details["dimension"] == 384


@pytest.mark.asyncio
async def test_probe_embeddings_failure(mock_config):
    with patch(
        "cortex.llm.client.embed_texts", side_effect=Exception("API key invalid")
    ):
        result = await probe_embeddings(mock_config)
        assert result.name == "Embeddings API"
        assert result.status == "fail"
        assert "API key invalid" in result.message


@pytest.mark.asyncio
async def test_check_reranker_success(mock_config):
    with patch("httpx.AsyncClient") as mock_client:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_client.return_value.__aenter__.return_value.get = AsyncMock(
            return_value=mock_response
        )
        result = await check_reranker(mock_config)
        assert result.name == "Reranker API"
        assert result.status == "pass"


@pytest.mark.asyncio
async def test_check_reranker_failure_is_warn(mock_config):
    with patch("httpx.AsyncClient") as mock_client:
        mock_client.return_value.__aenter__.return_value.get = AsyncMock(
            side_effect=httpx.ConnectError("Connection refused")
        )
        result = await check_reranker(mock_config)
        assert result.name == "Reranker API"
        assert result.status == "warn"
        assert "Cannot connect" in result.message


@pytest.mark.asyncio
async def test_check_reranker_not_configured(mock_config):
    mock_config.search.reranker_endpoint = None
    result = await check_reranker(mock_config)
    assert result.status == "pass"
    assert result.message == "Not configured"
