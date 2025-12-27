import unittest
from unittest.mock import MagicMock, patch

from cortex_cli._config_helpers import (
    _print_human_config,
    _print_json_config,
    _print_summary_sections,
)


class TestConfigHelpers(unittest.TestCase):
    def test_print_json_config_full(self):
        config = MagicMock()
        config.model_dump.return_value = {"foo": "bar"}
        with patch("builtins.print") as mock_print:
            _print_json_config(config)
            mock_print.assert_called()
            # Verify json output contains key
            args, _ = mock_print.call_args
            self.assertIn('"foo": "bar"', args[0])

    def test_print_json_config_section_found(self):
        config = MagicMock()
        config.model_dump.return_value = {"section1": {"foo": "bar"}}
        with patch("builtins.print") as mock_print:
            _print_json_config(config, section="section1")
            mock_print.assert_called()
            args, _ = mock_print.call_args
            self.assertIn('"section1":', args[0])

    def test_print_json_config_section_not_found(self):
        config = MagicMock()
        config.model_dump.return_value = {"key": "val"}
        with patch("builtins.print") as mock_print:
            _print_json_config(config, section="missing")
            mock_print.assert_called()
            args, _ = mock_print.call_args
            self.assertIn("Section 'missing' not found", args[0])

    def test_print_human_config_summary(self):
        config = MagicMock()
        # Mocking sub-configs
        config.core.env = "test"
        config.core.provider = "local"
        config.core.persona = "default"
        config.embedding.model_name = "emb-model"
        config.embedding.output_dimensionality = 768
        config.embedding.batch_size = 1
        config.embedding.embed_mode = "local"
        config.email.sender_locked_name = "tester"
        config.email.sender_locked_email = "test@test.com"
        config.email.reply_policy = "reply_all"
        config.search.fusion_strategy = "linear"
        config.search.k = 1
        config.search.recency_boost_strength = 0
        config.search.reranker_endpoint = ""
        config.processing.chunk_size = 512
        config.processing.chunk_overlap = 128

        with patch("builtins.print") as mock_print:
            _print_human_config(config)
            mock_print.assert_any_call("  Core")
            mock_print.assert_any_call("    Environment     test")

    def test_print_human_config_section(self):
        config = MagicMock()
        mock_do = MagicMock()
        mock_do.model_dump.return_value = {"key": "val"}
        config.digitalocean_llm = mock_do

        with patch("builtins.print") as mock_print:
            _print_human_config(config, section="DigitalOcean LLM")
            mock_print.assert_any_call("    key                  val")

    def test_print_human_config_unknown_section(self):
        config = MagicMock()
        # Ensure getattr returns None for 'unknown' attribute
        del config.unknown
        with patch("builtins.print") as mock_print:
            _print_human_config(config, section="unknown")
            # Only title should be printed, and an error
            self.assertEqual(mock_print.call_count, 2)
            args, _ = mock_print.call_args
            self.assertIn("Section 'unknown' not found", args[0])

    def test_print_summary_sections(self):
        config = MagicMock()

        # Mocking nested properties
        config.core.env = "test_env"
        config.core.provider = "test_provider"
        config.core.persona = "test_persona"
        config.embedding.model_name = "test_model"
        config.embedding.output_dimensionality = 128
        config.embedding.batch_size = 32
        config.embedding.embed_mode = "test_mode"
        config.email.sender_locked_name = "Test Sender"
        config.email.sender_locked_email = "sender@test.com"
        config.email.reply_policy = "reply_all"
        config.search.fusion_strategy = "linear"
        config.search.k = 10
        config.search.recency_boost_strength = 0.5
        config.search.reranker_endpoint = "http://rerank"
        config.processing.chunk_size = 1024
        config.processing.chunk_overlap = 256

        with patch("builtins.print") as mock_print:
            _print_summary_sections(config)
            mock_print.assert_any_call("  Core")
            mock_print.assert_any_call("    Environment     test_env")
            mock_print.assert_any_call("  Embeddings")
            mock_print.assert_any_call("    Model           test_model")
