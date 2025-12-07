"""DigitalOcean LLM scaling + inference helpers (Blueprint §§7, 17)."""
from __future__ import annotations

import logging
import math
import os
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin

import numpy as np
import requests
from cortex.common.exceptions import ConfigurationError, ProviderError
from cortex.config.models import DigitalOceanLLMConfig, DigitalOceanLLMModelConfig
from requests import Session
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

# Bytes per parameter for the supported quantisation schemes.
QUANT_BYTES: Dict[str, float] = {
    "fp16": 2.0,
    "bf16": 2.0,
    "fp32": 4.0,
    "fp8": 1.0,
    "int8": 1.0,
    "int4": 0.5,
    "int4_w4a16": 0.5,
    "nf4": 0.5,
}

DEFAULT_KV_CACHE_BYTES_PER_TOKEN: float = 131_072.0  # ~128KB/token


@dataclass
class ModelProfile:
    """Describe the hosted LLM for sizing the GPU pool."""

    name: str
    params_total: float
    context_length: int
    params_active: Optional[float] = None
    quantization: str = "fp16"
    additional_memory_gb: float = 4.0
    kv_bytes_per_token: float = DEFAULT_KV_CACHE_BYTES_PER_TOKEN
    tps_per_gpu: Optional[float] = None
    max_concurrent_requests_per_gpu: Optional[int] = None

    def __post_init__(self) -> None:
        if self.params_active is None:
            self.params_active = self.params_total

    def estimate_memory_gb(self) -> float:
        quant_bytes = QUANT_BYTES.get(self.quantization.lower())
        if quant_bytes is None:
            raise ValueError(f"Unknown quantization: {self.quantization}")

        weight_bytes = self.params_total * 1e9 * quant_bytes
        kv_bytes = float(self.context_length) * float(self.kv_bytes_per_token)
        total_bytes = weight_bytes + kv_bytes
        return total_bytes / (1024**3) + self.additional_memory_gb

    def concurrency_per_gpu(self) -> int:
        if (
            self.max_concurrent_requests_per_gpu
            and self.max_concurrent_requests_per_gpu > 0
        ):
            return int(self.max_concurrent_requests_per_gpu)
        return 1


def calculate_required_gpus(
    model: ModelProfile,
    memory_per_gpu_gb: float,
    headroom: float = 0.2,
) -> Tuple[int, float]:
    if memory_per_gpu_gb <= 0:
        raise ValueError("memory_per_gpu_gb must be > 0")
    if not (0.0 <= headroom < 1.0):
        raise ValueError("headroom must be in [0, 1)")

    total_mem_gb = model.estimate_memory_gb()
    usable_mem_gb = memory_per_gpu_gb * (1.0 - headroom)
    if usable_mem_gb <= 0:
        raise ValueError(f"Headroom {headroom} leaves no usable GPU memory")

    required_gpus = max(1, math.ceil(total_mem_gb / usable_mem_gb))
    return required_gpus, total_mem_gb / required_gpus


class DOApiClient:
    """Minimal DigitalOcean v2 API client with retries."""

    def __init__(
        self,
        token: Optional[str],
        base_url: str = "https://api.digitalocean.com/v2",
        timeout_s: int = 30,
        max_retries: int = 3,
        backoff_factor: float = 2.0,
        dry_run: bool = False,
    ) -> None:
        self.token = token or os.environ.get("DIGITALOCEAN_TOKEN", "")
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.dry_run = dry_run

        if not self.token and not self.dry_run:
            raise ConfigurationError("DigitalOcean API token is missing")

        self.session = Session()
        if not self.dry_run:
            self.session.headers.update(
                {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.token}",
                }
            )

        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=backoff_factor,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "POST", "PUT", "DELETE", "PATCH"],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    def request(self, method: str, path: str, **kwargs) -> Any:
        if self.dry_run and method.upper() != "GET":
            logger.info("[DRY RUN] %s %s %s", method, path, kwargs)
            return {"dry_run": True}

        url = f"{self.base_url}{path}"
        timeout = kwargs.pop("timeout", self.timeout_s)

        try:
            resp = self.session.request(method, url, timeout=timeout, **kwargs)
            if 400 <= resp.status_code < 500 and resp.status_code != 429:
                raise ProviderError(
                    f"DigitalOcean client error {resp.status_code}: {resp.text}",
                    provider="digitalocean",
                    retryable=False,
                )
            resp.raise_for_status()
        except requests.exceptions.RequestException as exc:
            raise ProviderError(
                f"DigitalOcean API unreachable: {exc}",
                provider="digitalocean",
                retryable=True,
            ) from exc

        if resp.status_code == 204 or not resp.content:
            return {}
        try:
            return resp.json()
        except ValueError:
            return {"raw": resp.text}

    def _paginate(self, path: str, key: str) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        next_url: Optional[str] = path

        while next_url:
            if next_url.startswith(self.base_url):
                next_url = next_url[len(self.base_url) :]

            data = self.request("GET", next_url)
            items.extend(data.get(key, []))

            links = data.get("links", {})
            next_url = links.get("pages", {}).get("next")

        return items

    def list_clusters(self) -> List[Dict[str, Any]]:
        return self._paginate("/kubernetes/clusters", "kubernetes_clusters")

    def get_cluster(self, cluster_id: str) -> Dict[str, Any]:
        return self.request("GET", f"/kubernetes/clusters/{cluster_id}")

    def create_cluster(
        self,
        name: str,
        region: str,
        version: Optional[str],
        node_size: str,
        count: int,
        tags: Optional[List[str]] = None,
        node_tags: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        if count <= 0:
            raise ValueError("Initial node count must be > 0")
        if not name or not region or not node_size:
            raise ValueError("name, region, and node_size are required")

        node_pool = {
            "name": f"{name}-gpu-pool",
            "size": node_size,
            "count": count,
            "auto_scale": False,
        }
        if node_tags:
            node_pool["tags"] = node_tags

        body: Dict[str, Any] = {
            "name": name,
            "region": region,
            "node_pools": [node_pool],
        }
        if version:
            body["version"] = version
        if tags:
            body["tags"] = tags

        return self.request("POST", "/kubernetes/clusters", json=body)

    def delete_cluster(self, cluster_id: str) -> Dict[str, Any]:
        return self.request("DELETE", f"/kubernetes/clusters/{cluster_id}")

    def list_node_pools(self, cluster_id: str) -> List[Dict[str, Any]]:
        return self._paginate(
            f"/kubernetes/clusters/{cluster_id}/node_pools", "node_pools"
        )

    def update_node_pool_count(
        self,
        cluster_id: str,
        node_pool_id: str,
        count: int,
        force_disable_autoscale: bool = False,
    ) -> Dict[str, Any]:
        if count < 0:
            raise ValueError("count must be >= 0")

        patch: Dict[str, Any] = {"count": count}
        if force_disable_autoscale:
            patch["auto_scale"] = False

        return self.request(
            "PATCH",
            f"/kubernetes/clusters/{cluster_id}/node_pools/{node_pool_id}",
            json=patch,
        )


class ClusterScaler:
    """Model-aware scaler for a GPU node pool."""

    def __init__(
        self,
        api_client: DOApiClient,
        memory_per_gpu_gb: float,
        gpus_per_node: int = 1,
        headroom: float = 0.2,
    ) -> None:
        if memory_per_gpu_gb <= 0:
            raise ValueError("memory_per_gpu_gb must be > 0")
        if gpus_per_node <= 0:
            raise ValueError("gpus_per_node must be > 0")
        if not (0.0 <= headroom < 1.0):
            raise ValueError("headroom must be in [0, 1)")

        self.api = api_client
        self.memory_per_gpu_gb = float(memory_per_gpu_gb)
        self.gpus_per_node = int(gpus_per_node)
        self.headroom = float(headroom)

    def plan_node_pool(
        self,
        model: ModelProfile,
        min_nodes: int = 0,
        max_nodes: int = 4,
    ) -> Tuple[int, int]:
        if min_nodes < 0:
            raise ValueError("min_nodes must be >= 0")
        if max_nodes <= 0:
            raise ValueError("max_nodes must be > 0")

        required_gpus, _ = calculate_required_gpus(
            model, self.memory_per_gpu_gb, headroom=self.headroom
        )
        required_nodes = max(1, math.ceil(required_gpus / self.gpus_per_node))

        initial = max(min_nodes, required_nodes)
        maximum = max(max_nodes, initial * 2)
        return initial, maximum

    def scale_to_match_load(
        self,
        cluster_id: str,
        node_pool_id: str,
        queued_requests: int,
        model: ModelProfile,
        min_nodes: int,
        max_nodes: int,
        max_scale_factor: float = 2.0,
        last_scale_time: Optional[float] = None,
        min_downscale_interval_s: int = 300,
        incoming_tokens_per_second: Optional[float] = None,
        target_gpu_utilization: float = 0.7,
    ) -> Dict[str, Any]:
        if queued_requests < 0:
            raise ValueError("queued_requests cannot be negative")
        if max_scale_factor <= 0:
            raise ValueError("max_scale_factor must be > 0")
        if min_downscale_interval_s < 0:
            raise ValueError("min_downscale_interval_s cannot be negative")
        if not (0.0 < target_gpu_utilization <= 1.0):
            raise ValueError("target_gpu_utilization must be in (0, 1]")
        if min_nodes < 0 or max_nodes <= 0 or min_nodes > max_nodes:
            raise ValueError("Invalid min_nodes/max_nodes combination")

        concurrency = model.concurrency_per_gpu()
        if queued_requests > 0:
            req_gpus_queue = max(1, math.ceil(queued_requests / concurrency))
        else:
            req_gpus_queue = 0

        req_gpus_tps = 0
        if incoming_tokens_per_second is not None and model.tps_per_gpu:
            if incoming_tokens_per_second > 0:
                eff_tps = model.tps_per_gpu * target_gpu_utilization
                req_gpus_tps = max(1, math.ceil(incoming_tokens_per_second / eff_tps))

        required_gpus = max(req_gpus_queue, req_gpus_tps)

        if required_gpus == 0:
            required_nodes = 0
        else:
            required_nodes = max(1, math.ceil(required_gpus / self.gpus_per_node))

        if self.api.dry_run:
            pool = {"id": node_pool_id, "count": max(min_nodes, 1), "auto_scale": False}
        else:
            pools = self.api.list_node_pools(cluster_id)
            pool = next((p for p in pools if p.get("id") == node_pool_id), None)
            if not pool:
                raise ProviderError(
                    f"Node pool {node_pool_id} not found in cluster {cluster_id}",
                    provider="digitalocean",
                    retryable=False,
                )

        current_count = int(pool.get("count", 0))
        is_autoscale_enabled = pool.get("auto_scale", False)

        hard_cap = (
            max_nodes
            if current_count == 0
            else int(max(current_count * max_scale_factor, current_count + 1))
        )
        desired = min(max(required_nodes, min_nodes), hard_cap, max_nodes)

        now = time.time()
        if desired < current_count and last_scale_time is not None:
            if now - last_scale_time < min_downscale_interval_s:
                if is_autoscale_enabled:
                    return self.api.update_node_pool_count(
                        cluster_id,
                        node_pool_id,
                        current_count,
                        force_disable_autoscale=True,
                    )
                return pool

        if desired == current_count:
            if is_autoscale_enabled:
                return self.api.update_node_pool_count(
                    cluster_id,
                    node_pool_id,
                    current_count,
                    force_disable_autoscale=True,
                )
            return pool

        return self.api.update_node_pool_count(
            cluster_id,
            node_pool_id,
            count=desired,
            force_disable_autoscale=is_autoscale_enabled,
        )


class DigitalOceanClusterManager:
    """Provision, scale, and destroy DigitalOcean GPU clusters."""

    def __init__(self, config: DigitalOceanLLMConfig) -> None:
        self.config = config
        self.scaling = config.scaling
        self.model = _model_profile_from_config(config.model)
        self.api = DOApiClient(
            token=self.scaling.token,
            base_url=self.scaling.api_base_url,
            dry_run=self.scaling.dry_run,
        )
        self.scaler = ClusterScaler(
            api_client=self.api,
            memory_per_gpu_gb=self.scaling.memory_per_gpu_gb,
            gpus_per_node=self.scaling.gpus_per_node,
            headroom=self.scaling.headroom,
        )

    def plan_gpu_pool(self) -> Tuple[int, int]:
        return self.scaler.plan_node_pool(
            self.model,
            min_nodes=self.scaling.min_nodes,
            max_nodes=self.scaling.max_nodes,
        )

    def provision_cluster(self) -> Dict[str, Any]:
        if not self.scaling.cluster_name:
            raise ConfigurationError(
                "digitalocean.scaling.cluster_name is required to provision"
            )
        if not self.scaling.region:
            raise ConfigurationError(
                "digitalocean.scaling.region is required to provision"
            )
        if not self.scaling.gpu_node_size:
            raise ConfigurationError(
                "digitalocean.scaling.gpu_node_size is required to provision"
            )

        initial, _ = self.plan_gpu_pool()
        node_count = max(initial, self.scaling.min_nodes or 0, 1)
        return self.api.create_cluster(
            name=self.scaling.cluster_name,
            region=self.scaling.region,
            version=self.scaling.kubernetes_version,
            node_size=self.scaling.gpu_node_size,
            count=node_count,
            tags=self.scaling.cluster_tags,
            node_tags=self.scaling.node_tags,
        )

    def destroy_cluster(self, cluster_id: Optional[str] = None) -> Dict[str, Any]:
        target = cluster_id or self.scaling.cluster_id
        if not target:
            raise ConfigurationError(
                "digitalocean.scaling.cluster_id is required to destroy a cluster"
            )
        return self.api.delete_cluster(target)

    def describe_cluster(self, cluster_id: Optional[str] = None) -> Dict[str, Any]:
        target = cluster_id or self.scaling.cluster_id
        if not target:
            raise ConfigurationError("cluster_id is required to describe a cluster")
        return self.api.get_cluster(target)


class DigitalOceanLLMService:
    """Manages scaling + inference calls to the DOKS-hosted LLM."""

    def __init__(self, config: DigitalOceanLLMConfig) -> None:
        self.scaling = config.scaling
        self.endpoint = config.endpoint
        self.model = _model_profile_from_config(config.model)

        self._api_client = DOApiClient(
            token=self.scaling.token,
            base_url=self.scaling.api_base_url,
            dry_run=self.scaling.dry_run,
        )
        self._scaler = ClusterScaler(
            api_client=self._api_client,
            memory_per_gpu_gb=self.scaling.memory_per_gpu_gb,
            gpus_per_node=self.scaling.gpus_per_node,
            headroom=self.scaling.headroom,
        )
        self._last_scale_time: Optional[float] = None
        self._inflight = 0
        self._inflight_lock = threading.Lock()
        self._http_session = self._build_session()

    def embed_texts(
        self, texts: List[str], expected_dim: Optional[int] = None
    ) -> np.ndarray:
        if not texts:
            dim = expected_dim or 0
            return np.zeros((0, dim), dtype=np.float32)

        payload = {
            "model": self.endpoint.default_embedding_model,
            "input": texts,
        }
        with self._tracked_request():
            response = self._post(self.endpoint.embedding_path, payload)
        vectors = self._parse_embeddings(response)
        return np.array(vectors, dtype=np.float32)

    def generate_text(
        self,
        prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        model: Optional[str] = None,
        response_format: Optional[Dict[str, Any]] = None,
        extra_payload: Optional[Dict[str, Any]] = None,
    ) -> str:
        payload: Dict[str, Any] = {
            "model": model or self.endpoint.default_completion_model,
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format:
            payload["response_format"] = response_format
        if extra_payload:
            payload.update(extra_payload)

        with self._tracked_request():
            response = self._post(self.endpoint.completion_path, payload)
        return self._extract_completion_text(response)

    def plan_node_pool(self) -> Tuple[int, int]:
        return self._scaler.plan_node_pool(
            self.model,
            min_nodes=self.scaling.min_nodes,
            max_nodes=self.scaling.max_nodes,
        )

    @contextmanager
    def _tracked_request(self):
        depth = self._increment_inflight()
        self._maybe_scale(depth)
        try:
            yield
        finally:
            self._decrement_inflight()

    def _increment_inflight(self) -> int:
        with self._inflight_lock:
            self._inflight += 1
            return self._inflight

    def _decrement_inflight(self) -> None:
        with self._inflight_lock:
            self._inflight = max(0, self._inflight - 1)

    def _maybe_scale(self, queued_requests: int) -> None:
        if not self.scaling.cluster_id or not self.scaling.node_pool_id:
            return
        try:
            self._scaler.scale_to_match_load(
                cluster_id=self.scaling.cluster_id,
                node_pool_id=self.scaling.node_pool_id,
                queued_requests=queued_requests,
                model=self.model,
                min_nodes=self.scaling.min_nodes,
                max_nodes=self.scaling.max_nodes,
                max_scale_factor=self.scaling.max_scale_factor,
                last_scale_time=self._last_scale_time,
                min_downscale_interval_s=self.scaling.min_downscale_interval_s,
                incoming_tokens_per_second=None,
                target_gpu_utilization=self.scaling.target_gpu_utilization,
            )
            self._last_scale_time = time.time()
        except ProviderError as exc:
            logger.warning("DigitalOcean scaling failed: %s", exc)

    def _build_session(self) -> Session:
        session = Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1.5,
            status_forcelist=[408, 409, 429, 500, 502, 503, 504],
            allowed_methods=["POST"],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        return session

    def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        if not self.endpoint.base_url:
            raise ConfigurationError(
                "digitalocean.endpoint.base_url is not configured",
                error_code="DO_LLM_ENDPOINT_MISSING",
            )

        url = urljoin(str(self.endpoint.base_url), path)
        headers = {"Content-Type": "application/json"}
        if self.endpoint.api_key:
            headers["Authorization"] = f"Bearer {self.endpoint.api_key}"
        if self.endpoint.extra_headers:
            headers.update(self.endpoint.extra_headers)

        try:
            resp = self._http_session.post(
                url,
                json=payload,
                timeout=self.endpoint.request_timeout_seconds,
                verify=self.endpoint.verify_tls,
            )
            if resp.status_code >= 400:
                raise ProviderError(
                    f"DigitalOcean LLM gateway error {resp.status_code}: {resp.text}",
                    provider="digitalocean",
                    retryable=resp.status_code >= 500,
                )
            return resp.json()
        except requests.exceptions.RequestException as exc:
            raise ProviderError(
                f"DigitalOcean LLM gateway unreachable: {exc}",
                provider="digitalocean",
                retryable=True,
            ) from exc
        except ValueError as exc:
            raise ProviderError(
                f"DigitalOcean LLM gateway returned invalid JSON: {exc}",
                provider="digitalocean",
                retryable=False,
            ) from exc

    @staticmethod
    def _extract_completion_text(payload: Dict[str, Any]) -> str:
        if not payload:
            raise ProviderError(
                "Empty completion response", provider="digitalocean", retryable=True
            )

        choices = payload.get("choices")
        if isinstance(choices, list) and choices:
            choice = choices[0]
            if isinstance(choice, dict):
                if choice.get("text"):
                    return str(choice["text"])
                message = choice.get("message", {})
                content = message.get("content") if isinstance(message, dict) else None
                if isinstance(content, str):
                    return content
                if isinstance(content, list):
                    return "".join(
                        str(part.get("text", ""))
                        for part in content
                        if isinstance(part, dict)
                    )
        if payload.get("output"):
            return str(payload["output"])
        raise ProviderError(
            "Unable to extract completion text from gateway response",
            provider="digitalocean",
            retryable=False,
        )

    @staticmethod
    def _parse_embeddings(payload: Dict[str, Any]) -> List[List[float]]:
        data = payload.get("data")
        if not isinstance(data, list) or not data:
            raise ProviderError(
                "Embedding payload missing data",
                provider="digitalocean",
                retryable=False,
            )

        vectors: List[List[float]] = []
        for row in data:
            if not isinstance(row, dict):
                continue
            vector = row.get("embedding") or row.get("vector")
            if not isinstance(vector, list):
                raise ProviderError(
                    "Embedding row missing vector",
                    provider="digitalocean",
                    retryable=False,
                )
            vectors.append([float(v) for v in vector])
        return vectors


def _model_profile_from_config(
    model_config: DigitalOceanLLMModelConfig,
) -> ModelProfile:
    return ModelProfile(
        name=model_config.name,
        params_total=model_config.params_total,
        params_active=model_config.params_active,
        context_length=model_config.context_length,
        quantization=model_config.quantization,
        additional_memory_gb=model_config.additional_memory_gb,
        kv_bytes_per_token=model_config.kv_bytes_per_token,
        tps_per_gpu=model_config.tps_per_gpu,
        max_concurrent_requests_per_gpu=model_config.max_concurrent_requests_per_gpu,
    )
