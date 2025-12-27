import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parents[2]
sys.path.append(str(root_dir / "backend" / "src"))

import unittest  # noqa: E402
from unittest.mock import patch  # noqa: E402

from cortex.llm.runtime import LLMRuntime, get_runtime  # noqa: E402
from cortex.orchestration.graphs import (  # noqa: E402
    build_answer_graph,
    build_draft_graph,
)
from cortex.retrieval.hybrid_search import KBSearchInput  # noqa: E402


class TestIntelligenceRuntime(unittest.TestCase):
    def test_llm_runtime_initialization(self):
        print("\nTesting LLM Runtime initialization...")
        runtime = get_runtime()
        assert isinstance(runtime, LLMRuntime)
        assert runtime.retry_config is not None
        print("PASS: LLM Runtime initialized")

    @patch("cortex.llm.runtime.VLLMProvider")
    def test_llm_runtime_mock_completion(self, mock_vllm_provider):
        print("\nTesting LLM Runtime (Mock)...")
        # Configure the mock provider's complete method
        mock_provider_instance = mock_vllm_provider.return_value
        mock_provider_instance.complete.return_value = "Mock response"

        # We need to get a new runtime instance to ensure it uses the patched provider
        runtime = LLMRuntime()

        # Execute the completion and assert the result
        res = runtime.complete_text("Hello")
        assert res == "Mock response"

        # Verify that the mock was called correctly
        mock_provider_instance.complete.assert_called_once_with("Hello")
        print("PASS: LLM completion (mocked)")

    def test_orchestration_graphs(self):
        print("\nTesting Orchestration Graphs...")
        graph_answer = build_answer_graph()
        assert graph_answer is not None
        print("PASS: Answer Graph built")

        graph_draft = build_draft_graph()
        assert graph_draft is not None
        print("PASS: Draft Graph built")

    def test_retrieval_models(self):
        print("\nTesting Retrieval Models...")
        input_data = {
            "tenant_id": "test",
            "user_id": "user1",
            "query": "test query",
            "fusion_method": "rrf",
        }
        model = KBSearchInput(**input_data)
        assert model.query == "test query"
        print("PASS: KBSearchInput validated")


if __name__ == "__main__":
    unittest.main()
