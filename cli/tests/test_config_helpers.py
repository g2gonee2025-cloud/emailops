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
        config.embedding.model_name = "emb-model"
        # etc...
        with patch("builtins.print") as mock_print:
            _print_human_config(config)
            mock_print.assert_called()

    def test_print_human_config_section(self):
        config = MagicMock()
        config.digitalocean_llm.model_dump.return_value = {"k": "v"}
        with patch("builtins.print") as mock_print:
            _print_human_config(config, section="DigitalOcean LLM")
            mock_print.assert_called()

    def test_print_human_config_unknown_section(self):
        config = MagicMock()
        # Ensure getattr returns None
        del config.unknown
        with patch("builtins.print") as mock_print:
            # Should produce no output beyond header potentially, or handle gracefully
            _print_human_config(config, section="unknown")
            mock_print.assert_called()

    def test_print_summary_sections(self):
        config = MagicMock()
        with patch("builtins.print") as mock_print:
            _print_summary_sections(config)
            mock_print.assert_called()
