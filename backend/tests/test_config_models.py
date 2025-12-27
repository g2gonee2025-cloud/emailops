import os
from unittest.mock import patch

from cortex.config.models import (
    RetryConfig,
    SensitiveConfig,
    SummarizerConfig,
)


def test_sensitive_config_huggingface_api_key():
    """Test that SensitiveConfig correctly retrieves the HuggingFace API key."""
    api_key = "test_api_key"
    with patch.dict(os.environ, {"OUTLOOKCORTEX_HUGGINGFACE_API_KEY": api_key}):
        config = SensitiveConfig()
        assert config.huggingface_api_key == api_key


def test_retry_config_api_max_retries():
    """Test that RetryConfig correctly retrieves the API max retries value."""
    max_retries = "5"
    with patch.dict(os.environ, {"OUTLOOKCORTEX_API_MAX_RETRIES": max_retries}):
        config = RetryConfig()
        assert config.max_retries == int(max_retries)


def test_summarizer_config_summarizer_version():
    """Test that SummarizerConfig correctly retrieves the summarizer version."""
    summarizer_version = "test_version"
    with patch.dict(
        os.environ, {"OUTLOOKCORTEX_SUMMARIZER_VERSION": summarizer_version}
    ):
        config = SummarizerConfig()
        assert config.summarizer_version == summarizer_version
