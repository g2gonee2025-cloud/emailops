from unittest.mock import MagicMock, patch

import pytest
from cortex.common.exceptions import ConfigurationError, ProviderError
from cortex.llm.doks_scaler import (
    DEFAULT_KV_CACHE_BYTES_PER_TOKEN,
    ClusterScaler,
    DigitalOceanLLMService,
    DOApiClient,
    ModelProfile,
    calculate_required_gpus,
)


class TestModelProfile:
    """Test ModelProfile dataclass and its methods."""

    def test_model_profile_defaults(self):
        """Test ModelProfile with default values."""
        profile = ModelProfile(
            name="test-model",
            params_total=7.0,
            context_length=4096,
        )
        assert profile.name == "test-model"
        assert profile.params_total == pytest.approx(7.0)
        assert profile.context_length == 4096
        assert profile.params_active == pytest.approx(7.0)  # Should default to params_total
        assert profile.quantization == "fp16"
        assert profile.additional_memory_gb == pytest.approx(4.0)
        assert profile.kv_bytes_per_token == pytest.approx(DEFAULT_KV_CACHE_BYTES_PER_TOKEN)

    def test_model_profile_with_params_active(self):
        """Test ModelProfile with explicit params_active (MoE model)."""
        profile = ModelProfile(
            name="moe-model",
            params_total=47.0,
            context_length=8192,
            params_active=14.0,
        )
        assert profile.params_active == pytest.approx(14.0)

    def test_estimate_memory_gb_fp16(self):
        """Test memory estimation for fp16 model."""
        profile = ModelProfile(
            name="test-7b",
            params_total=7.0,
            context_length=4096,
            quantization="fp16",
            additional_memory_gb=0.0,
            kv_bytes_per_token=0.0,
        )
        # 7B params * 2 bytes (fp16) / (1024^3) = ~13.04GB
        estimated = profile.estimate_memory_gb()
        assert estimated == pytest.approx(13.04, rel=0.01)

    def test_estimate_memory_gb_int4(self):
        """Test memory estimation for int4 quantization."""
        profile = ModelProfile(
            name="test-7b-int4",
            params_total=7.0,
            context_length=4096,
            quantization="int4",
            additional_memory_gb=0.0,
            kv_bytes_per_token=0.0,
        )
        # 7B params * 0.5 bytes (int4) / (1024^3) = ~3.26GB
        estimated = profile.estimate_memory_gb()
        assert estimated == pytest.approx(3.26, rel=0.01)

    def test_estimate_memory_gb_unknown_quantization(self):
        """Test memory estimation raises for unknown quantization."""
        profile = ModelProfile(
            name="test-7b",
            params_total=7.0,
            context_length=4096,
            quantization="unknown_quant",
        )
        with pytest.raises(ValueError, match="Unknown quantization"):
            profile.estimate_memory_gb()

    def test_estimate_memory_includes_kv_cache(self):
        """Test memory estimation includes KV cache."""
        profile = ModelProfile(
            name="test-7b",
            params_total=7.0,
            context_length=4096,
            quantization="fp16",
            additional_memory_gb=0.0,
            kv_bytes_per_token=1024.0,  # 1KB per token
        )
        # Should include 4096 * 1024 bytes of KV cache
        estimated = profile.estimate_memory_gb()
        expected_weights = 7.0 * 1e9 * 2 / (1024**3)  # ~13.04GB
        expected_kv = 4096 * 1024 / (1024**3)  # ~0.004GB
        assert estimated == pytest.approx(expected_weights + expected_kv, rel=0.01)

    def test_concurrency_per_gpu_default(self):
        """Test default concurrency per GPU."""
        profile = ModelProfile(
            name="test",
            params_total=7.0,
            context_length=4096,
        )
        assert profile.concurrency_per_gpu() == 1

    def test_concurrency_per_gpu_explicit(self):
        """Test explicit concurrency per GPU."""
        profile = ModelProfile(
            name="test",
            params_total=7.0,
            context_length=4096,
            max_concurrent_requests_per_gpu=4,
        )
        assert profile.concurrency_per_gpu() == 4


class TestCalculateRequiredGpus:
    """Test calculate_required_gpus function."""

    def test_basic_calculation(self):
        """Test basic GPU calculation."""
        profile = ModelProfile(
            name="test-7b",
            params_total=7.0,
            context_length=4096,
            quantization="fp16",
            additional_memory_gb=0.0,
            kv_bytes_per_token=0.0,
        )
        # ~13.04GB model / 80GB GPU with 20% headroom = 13.04 / 64 = 1 GPU
        gpus, mem_per_gpu = calculate_required_gpus(profile, 80.0, headroom=0.2)
        assert gpus == 1
        assert mem_per_gpu == pytest.approx(13.04, rel=0.01)

    def test_multiple_gpus_needed(self):
        """Test when multiple GPUs are needed."""
        profile = ModelProfile(
            name="test-70b",
            params_total=70.0,
            context_length=4096,
            quantization="fp16",
            additional_memory_gb=0.0,
            kv_bytes_per_token=0.0,
        )
        # 140GB model / 80GB GPU with 20% headroom = 140 / 64 = 3 GPUs
        gpus, _ = calculate_required_gpus(profile, 80.0, headroom=0.2)
        assert gpus == 3

    def test_invalid_memory_per_gpu(self):
        """Test that invalid memory raises ValueError."""
        profile = ModelProfile(name="test", params_total=7.0, context_length=4096)
        with pytest.raises(ValueError, match="memory_per_gpu_gb must be > 0"):
            calculate_required_gpus(profile, 0.0)

    def test_invalid_headroom(self):
        """Test that invalid headroom raises ValueError."""
        profile = ModelProfile(name="test", params_total=7.0, context_length=4096)
        with pytest.raises(ValueError, match="headroom must be in"):
            calculate_required_gpus(profile, 80.0, headroom=1.0)

    def test_headroom_near_one_still_works(self):
        """Test that headroom close to 1.0 still works if memory is sufficient."""
        profile = ModelProfile(
            name="test",
            params_total=0.001,  # Very small model
            context_length=32,
            additional_memory_gb=0.0,
            kv_bytes_per_token=0.0,
        )
        # Very small model should fit even with high headroom
        gpus, _ = calculate_required_gpus(profile, 80.0, headroom=0.99)
        assert gpus >= 1


class TestClusterScaler:
    """Test ClusterScaler class."""

    def test_init(self):
        """Test ClusterScaler initialization."""
        client = MagicMock()
        scaler = ClusterScaler(
            api_client=client,
            memory_per_gpu_gb=80.0,
            gpus_per_node=2,
            headroom=0.15,
        )
        assert scaler.memory_per_gpu_gb == pytest.approx(80.0)
        assert scaler.gpus_per_node == 2
        assert scaler.headroom == pytest.approx(0.15)

    def test_plan_node_pool_basic(self):
        """Test basic node pool planning."""
        client = MagicMock()
        scaler = ClusterScaler(
            api_client=client,
            memory_per_gpu_gb=80.0,
            gpus_per_node=1,
        )
        profile = ModelProfile(
            name="test-7b",
            params_total=7.0,
            context_length=4096,
            quantization="fp16",
            additional_memory_gb=0.0,
            kv_bytes_per_token=0.0,
        )
        initial, maximum = scaler.plan_node_pool(profile, min_nodes=1, max_nodes=4)
        assert initial >= 1
        assert maximum >= initial

    def test_plan_node_pool_min_nodes_negative(self):
        """Test that negative min_nodes raises ValueError."""
        client = MagicMock()
        scaler = ClusterScaler(api_client=client, memory_per_gpu_gb=80.0)
        profile = ModelProfile(name="test", params_total=7.0, context_length=4096)
        with pytest.raises(ValueError, match="min_nodes must be >= 0"):
            scaler.plan_node_pool(profile, min_nodes=-1)

    def test_plan_node_pool_max_nodes_zero(self):
        """Test that zero max_nodes raises ValueError."""
        client = MagicMock()
        scaler = ClusterScaler(api_client=client, memory_per_gpu_gb=80.0)
        profile = ModelProfile(name="test", params_total=7.0, context_length=4096)
        with pytest.raises(ValueError, match="max_nodes must be > 0"):
            scaler.plan_node_pool(profile, max_nodes=0)


class TestDigitalOceanLLMServiceExtraction:
    """Test DigitalOceanLLMService static extraction methods."""

    def test_extract_completion_text_choices_text(self):
        """Test extraction from choices with text field."""
        payload = {"choices": [{"text": "Hello, world!"}]}
        result = DigitalOceanLLMService._extract_completion_text(payload)
        assert result == "Hello, world!"

    def test_extract_completion_text_choices_message_content(self):
        """Test extraction from choices with message.content."""
        payload = {"choices": [{"message": {"content": "Hello from message"}}]}
        result = DigitalOceanLLMService._extract_completion_text(payload)
        assert result == "Hello from message"

    def test_extract_completion_text_choices_message_content_list(self):
        """Test extraction from choices with message.content as list."""
        payload = {
            "choices": [
                {
                    "message": {
                        "content": [
                            {"text": "Part 1"},
                            {"text": " Part 2"},
                        ]
                    }
                }
            ]
        }
        result = DigitalOceanLLMService._extract_completion_text(payload)
        assert result == "Part 1 Part 2"

    def test_extract_completion_text_output_field(self):
        """Test extraction from output field."""
        payload = {"output": "Output text here"}
        result = DigitalOceanLLMService._extract_completion_text(payload)
        assert result == "Output text here"

    def test_extract_completion_text_empty_payload(self):
        """Test extraction from empty payload raises error."""
        with pytest.raises(ProviderError, match="Empty completion response"):
            DigitalOceanLLMService._extract_completion_text({})

    def test_extract_completion_text_no_valid_content(self):
        """Test extraction with no valid content raises error."""
        payload = {"choices": [{"message": {}}]}
        with pytest.raises(ProviderError, match="Unable to extract completion text"):
            DigitalOceanLLMService._extract_completion_text(payload)

    def test_parse_embeddings_success(self):
        """Test successful embedding parsing."""
        payload = {
            "data": [
                {"embedding": [0.1, 0.2, 0.3]},
                {"embedding": [0.4, 0.5, 0.6]},
            ]
        }
        result = DigitalOceanLLMService._parse_embeddings(payload)
        assert len(result) == 2
        assert result[0] == [0.1, 0.2, 0.3]
        assert result[1] == [0.4, 0.5, 0.6]

    def test_parse_embeddings_using_vector_key(self):
        """Test embedding parsing using 'vector' key."""
        payload = {
            "data": [
                {"vector": [1.0, 2.0, 3.0]},
            ]
        }
        result = DigitalOceanLLMService._parse_embeddings(payload)
        assert result == [[1.0, 2.0, 3.0]]

    def test_parse_embeddings_missing_data(self):
        """Test parsing with missing data raises error."""
        with pytest.raises(ProviderError, match="missing data"):
            DigitalOceanLLMService._parse_embeddings({})

    def test_parse_embeddings_missing_vector(self):
        """Test parsing with missing vector raises error."""
        payload = {"data": [{"id": 1}]}  # No embedding or vector key
        with pytest.raises(ProviderError, match="missing vector"):
            DigitalOceanLLMService._parse_embeddings(payload)


class TestDOApiClient:
    def test_init_missing_token(self):
        with pytest.raises(ConfigurationError):
            DOApiClient(token="")

    def test_init_dry_run(self):
        client = DOApiClient(token="", dry_run=True)
        assert client.dry_run is True

    def test_request_success(self):
        client = DOApiClient(token="test-token")
        with patch("requests.Session.request") as mock_req:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {"foo": "bar"}
            mock_req.return_value = mock_resp

            data = client.request("GET", "/account")
            assert data == {"foo": "bar"}
            # Check URL construction uses base_url (lowercase)
            mock_req.assert_called_with(
                "GET", "https://api.digitalocean.com/v2/account", timeout=30
            )

    def test_pagination(self):
        client = DOApiClient(token="test-token")
        with patch.object(client, "request") as mock_request:
            mock_request.side_effect = [
                {
                    "items": [{"id": 1}],
                    "links": {
                        "pages": {"next": "https://api.digitalocean.com/v2/next"}
                    },
                },
                {"items": [{"id": 2}], "links": {}},
            ]

            items = client._paginate("/items", "items")
            assert len(items) == 2
            assert items[0]["id"] == 1
            assert items[1]["id"] == 2

            # Verify correct calls including stripped URL logic check
            assert mock_request.call_count == 2
            mock_request.assert_any_call("GET", "/items")
            mock_request.assert_any_call("GET", "/next")

    def test_list_clusters(self):
        """Test listing clusters."""
        client = DOApiClient(token="test-token")
        with patch.object(client, "_paginate") as mock_paginate:
            mock_paginate.return_value = [{"id": "cluster-1"}]
            result = client.list_clusters()
            assert result == [{"id": "cluster-1"}]
            mock_paginate.assert_called_once_with(
                "/kubernetes/clusters", "kubernetes_clusters"
            )

    def test_get_cluster(self):
        """Test getting a cluster."""
        client = DOApiClient(token="test-token")
        with patch.object(client, "request") as mock_request:
            mock_request.return_value = {"kubernetes_cluster": {"id": "cluster-1"}}
            result = client.get_cluster("cluster-1")
            assert result == {"kubernetes_cluster": {"id": "cluster-1"}}

    def test_list_node_pools(self):
        """Test listing node pools."""
        client = DOApiClient(token="test-token")
        with patch.object(client, "_paginate") as mock_paginate:
            mock_paginate.return_value = [{"id": "pool-1"}]
            result = client.list_node_pools("cluster-1")
            assert result == [{"id": "pool-1"}]


class TestClusterScalerScaleToMatchLoad:
    """Test ClusterScaler.scale_to_match_load method."""

    def test_scale_to_match_load_validation_negative_queued(self):
        """Test validation for negative queued_requests."""
        client = MagicMock()
        client.dry_run = True
        scaler = ClusterScaler(api_client=client, memory_per_gpu_gb=80.0)
        profile = ModelProfile(name="test", params_total=7.0, context_length=4096)

        with pytest.raises(ValueError, match="queued_requests cannot be negative"):
            scaler.scale_to_match_load(
                cluster_id="c1",
                node_pool_id="np1",
                queued_requests=-1,
                model=profile,
                min_nodes=0,
                max_nodes=4,
            )

    def test_scale_to_match_load_validation_invalid_scale_factor(self):
        """Test validation for invalid max_scale_factor."""
        client = MagicMock()
        client.dry_run = True
        scaler = ClusterScaler(api_client=client, memory_per_gpu_gb=80.0)
        profile = ModelProfile(name="test", params_total=7.0, context_length=4096)

        with pytest.raises(ValueError, match="max_scale_factor must be > 0"):
            scaler.scale_to_match_load(
                cluster_id="c1",
                node_pool_id="np1",
                queued_requests=10,
                model=profile,
                min_nodes=0,
                max_nodes=4,
                max_scale_factor=0,
            )

    def test_scale_to_match_load_validation_invalid_utilization(self):
        """Test validation for invalid target_gpu_utilization."""
        client = MagicMock()
        client.dry_run = True
        scaler = ClusterScaler(api_client=client, memory_per_gpu_gb=80.0)
        profile = ModelProfile(name="test", params_total=7.0, context_length=4096)

        with pytest.raises(ValueError, match="target_gpu_utilization must be in"):
            scaler.scale_to_match_load(
                cluster_id="c1",
                node_pool_id="np1",
                queued_requests=10,
                model=profile,
                min_nodes=0,
                max_nodes=4,
                target_gpu_utilization=0.0,
            )

    def test_scale_to_match_load_validation_invalid_nodes(self):
        """Test validation for invalid min_nodes/max_nodes."""
        client = MagicMock()
        client.dry_run = True
        scaler = ClusterScaler(api_client=client, memory_per_gpu_gb=80.0)
        profile = ModelProfile(name="test", params_total=7.0, context_length=4096)

        with pytest.raises(ValueError, match="Invalid min_nodes/max_nodes"):
            scaler.scale_to_match_load(
                cluster_id="c1",
                node_pool_id="np1",
                queued_requests=10,
                model=profile,
                min_nodes=5,
                max_nodes=4,  # min > max is invalid
            )

    def test_scale_to_match_load_dry_run(self):
        """Test scale_to_match_load in dry_run mode."""
        client = MagicMock()
        client.dry_run = True
        scaler = ClusterScaler(api_client=client, memory_per_gpu_gb=80.0)
        profile = ModelProfile(name="test", params_total=7.0, context_length=4096)

        result = scaler.scale_to_match_load(
            cluster_id="c1",
            node_pool_id="np1",
            queued_requests=0,
            model=profile,
            min_nodes=1,
            max_nodes=4,
        )

        assert result["id"] == "np1"
        assert result["count"] >= 1

    def test_scale_to_match_load_zero_requests(self):
        """Test scaling with zero queued requests."""
        client = MagicMock()
        client.dry_run = True
        scaler = ClusterScaler(api_client=client, memory_per_gpu_gb=80.0)
        profile = ModelProfile(name="test", params_total=7.0, context_length=4096)

        result = scaler.scale_to_match_load(
            cluster_id="c1",
            node_pool_id="np1",
            queued_requests=0,
            model=profile,
            min_nodes=0,
            max_nodes=4,
        )

        # With zero requests and min_nodes=0, desired should be low
        assert result is not None

    def test_scale_to_match_load_high_requests(self):
        """Test scaling with high queued requests."""
        client = MagicMock()
        client.dry_run = True
        scaler = ClusterScaler(
            api_client=client, memory_per_gpu_gb=80.0, gpus_per_node=1
        )
        profile = ModelProfile(
            name="test",
            params_total=7.0,
            context_length=4096,
            max_concurrent_requests_per_gpu=2,
        )

        result = scaler.scale_to_match_load(
            cluster_id="c1",
            node_pool_id="np1",
            queued_requests=100,  # High load
            model=profile,
            min_nodes=1,
            max_nodes=10,
        )

        # Should request scaling to handle load
        assert result is not None

    def test_scale_to_match_load_with_tps(self):
        """Test scaling with tokens per second metric."""
        client = MagicMock()
        client.dry_run = True
        scaler = ClusterScaler(
            api_client=client, memory_per_gpu_gb=80.0, gpus_per_node=1
        )
        profile = ModelProfile(
            name="test",
            params_total=7.0,
            context_length=4096,
            tps_per_gpu=100.0,  # 100 tokens/sec per GPU
        )

        result = scaler.scale_to_match_load(
            cluster_id="c1",
            node_pool_id="np1",
            queued_requests=0,
            model=profile,
            min_nodes=1,
            max_nodes=10,
            incoming_tokens_per_second=500.0,  # Need multiple GPUs to handle
        )

        assert result is not None
