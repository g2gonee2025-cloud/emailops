import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.append(str(Path("backend/src").resolve()))

from cortex.llm import runtime


@pytest.fixture(autouse=True)
def reset_runtime_singleton(monkeypatch):
    """Ensure each test gets a fresh runtime instance."""
    monkeypatch.setattr(runtime, "_runtime_instance", None)
    yield
    monkeypatch.setattr(runtime, "_runtime_instance", None)


def test_get_runtime_exposes_retry_config():
    rt = runtime.get_runtime()
    assert isinstance(rt, runtime.LLMRuntime)
    assert getattr(rt, "retry_config") is not None


def test_complete_text_ignores_nonstring_reasoning():
    with patch("cortex.llm.runtime._config") as mock_config, patch(
        "openai.OpenAI"
    ) as mock_openai:
        mock_config.core.provider = "openai"
        mock_config.sensitive.openai_api_key = "sk-mock"

        retry_mock = MagicMock()
        retry_mock.max_retries = 2
        retry_mock.rate_limit_capacity = 10
        retry_mock.rate_limit_per_sec = 5
        retry_mock.circuit_failure_threshold = 3
        retry_mock.circuit_reset_seconds = 30
        mock_config.retry = retry_mock

        rt = runtime.LLMRuntime()

        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Mock response"
        mock_client.chat.completions.create.return_value = mock_response

        assert rt.complete_text("Hello") == "Mock response"
