import os
import sys
from pathlib import Path


def find_project_root(marker="pyproject.toml"):
    """Find the project root by searching for a marker file."""
    current_path = Path(__file__).resolve()
    while current_path != current_path.parent:
        if (current_path / marker).exists():
            return current_path
        current_path = current_path.parent
    raise FileNotFoundError(f"Project root with marker '{marker}' not found.")


# Add the project's 'src' directory to the Python path
try:
    root_dir = find_project_root()
    src_path = root_dir / "backend" / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
except FileNotFoundError as e:
    print(f"Error: {e}", file=sys.stderr)
    sys.exit(1)

import unittest
from unittest.mock import patch

from cortex.llm.runtime import LLMRuntime, get_runtime
from cortex.orchestration.graphs import (
    build_answer_graph,
    build_draft_graph,
)
from cortex.retrieval.hybrid_search import KBSearchInput


class TestIntelligenceRuntime(unittest.TestCase):
    @patch("cortex.llm.runtime.VLLMProvider")
    def test_llm_runtime_initialization(self, mock_vllm_provider):
        with self.assertLogs("cortex.llm.runtime", level="INFO"):
            runtime = get_runtime()
            self.assertIsInstance(runtime, LLMRuntime)
            self.assertIsNotNone(runtime.retry_config)

    @patch("cortex.llm.runtime.VLLMProvider")
    def test_llm_runtime_mock_completion(self, mock_vllm_provider):
        # Configure the mock provider's complete method
        mock_provider_instance = mock_vllm_provider.return_value
        mock_provider_instance.complete.return_value = "Mock response"

        # We need to get a new runtime instance to ensure it uses the patched provider
        runtime = LLMRuntime()

        # Execute the completion and assert the result
        res = runtime.complete_text("Hello")
        self.assertEqual(res, "Mock response")

        # Verify that the mock was called correctly
        mock_provider_instance.complete.assert_called_once()
        args, _ = mock_provider_instance.complete.call_args
        expected_message = [{"role": "user", "content": "Hello"}]
        self.assertEqual(args[0], expected_message)

    def test_orchestration_graphs(self):
        graph_answer = build_answer_graph()
        self.assertIsNotNone(graph_answer)

        graph_draft = build_draft_graph()
        self.assertIsNotNone(graph_draft)

    def test_retrieval_models(self):
        input_data = {
            "tenant_id": "test",
            "user_id": "user1",
            "query": "test query",
            "fusion_method": "rrf",
        }
        model = KBSearchInput(**input_data)
        self.assertEqual(model.query, "test query")


if __name__ == "__main__":
    unittest.main()
