import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parents[2]
sys.path.append(str(root_dir / "backend" / "src"))
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.append(str(Path("backend/src").resolve()))

from cortex.config.loader import get_config
from cortex.llm.runtime import LLMRuntime, get_runtime
from cortex.orchestration.graphs import build_answer_graph, build_draft_graph
from cortex.retrieval.hybrid_search import KBSearchInput


class TestIntelligenceRuntime(unittest.TestCase):
    def setUp(self):
        self.config = get_config()

    def test_llm_runtime_initialization(self):
        print("\nTesting LLM Runtime initialization...")
        runtime = get_runtime()
        assert isinstance(runtime, LLMRuntime)
        assert runtime.retry_config is not None
        print("PASS: LLM Runtime initialized")

    @patch("cortex.llm.runtime._config")
    def test_llm_runtime_mock_completion(self, mock_config):
        print("\nTesting LLM Runtime (Mock)...")
        # Mock config to avoid real provider calls
        mock_config.core.provider = "openai"
        mock_config.sensitive.openai_api_key = "sk-mock"

        # Explicitly configure retry mock to ensure int values
        retry_mock = MagicMock()
        retry_mock.max_retries = 3
        retry_mock.rate_limit_capacity = 100
        retry_mock.rate_limit_per_sec = 10
        retry_mock.circuit_failure_threshold = 5
        retry_mock.circuit_reset_seconds = 60
        mock_config.retry = retry_mock

        runtime = LLMRuntime()
        with patch("openai.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client

            mock_response = MagicMock()
            mock_response.choices[0].message.content = "Mock response"
            mock_client.chat.completions.create.return_value = mock_response

            res = runtime.complete_text("Hello")
            assert res == "Mock response"
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
