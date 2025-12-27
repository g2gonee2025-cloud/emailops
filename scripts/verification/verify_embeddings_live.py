import os
import time
from urllib.parse import urlparse

import numpy as np
import requests

# Configuration
DEFAULT_BASE_URL = "http://localhost:8081"
_env_url = os.getenv("DO_LLM_BASE_URL", "").strip()
if _env_url:
    _parsed = urlparse(_env_url)
    if _parsed.scheme in ("http", "https") and _parsed.netloc:
        BASE_URL = _env_url.rstrip("/")
    else:
        BASE_URL = DEFAULT_BASE_URL
else:
    BASE_URL = DEFAULT_BASE_URL
MODEL_ID = "tencent/KaLM-Embedding-Gemma3-12B-2511"
EXPECTED_DIM = 3840


def test_live_embeddings() -> None:
    print(f"Testing Embeddings API at: {BASE_URL}")

    # vLLM uses OpenAI-compatible API format
    payload = {
        "model": MODEL_ID,
        "input": ["Hello world", "This is a test of the emailops embedding service."],
    }

    try:
        start_time = time.time()
        response = requests.post(f"{BASE_URL}/v1/embeddings", json=payload)
        response.raise_for_status()
        duration = time.time() - start_time

        # vLLM returns OpenAI-compatible format
        data = response.json()
        # Safely extract embeddings; handle cases where "data" may be missing
        if not isinstance(data, dict):
            raise ValueError("Unexpected response format: expected a JSON object.")
        embeddings = []
        if "data" in data and isinstance(data["data"], list):
            items = data.get("data") if isinstance(data, dict) else []
            if not isinstance(items, list):
                items = []
            for idx, item in enumerate(items):
                if (
                    not isinstance(item, dict)
                    or "embedding" not in item
                    or item["embedding"] is None
                ):
                    print(
                        f"⚠️ Skipping item at index {idx} without a valid 'embedding' field"
                    )
                    continue
                embeddings.append(item["embedding"])
        elif "embedding" in data:
            embeddings = [data["embedding"]]
        elif "embeddings" in data and isinstance(data["embeddings"], list):
            embeddings = data["embeddings"]
        if not embeddings:
            raise ValueError("No embeddings found in the response.")

        print(f"✅ Request successful in {duration:.2f}s")
        print(f"Received {len(embeddings)} embeddings")

        # Validate dimensions
        if not embeddings:
            print("❌ No embeddings returned")
            dim = 0
        else:
            dim = len(embeddings[0])
        print(f"Vector Dimension: {dim}")

        if dim == EXPECTED_DIM:
            print(f"✅ Dimension matches expected {EXPECTED_DIM}")
        else:
            print(f"❌ Dimension mismatch! Expected {EXPECTED_DIM}, got {dim}")
            exit(1)

        # Validate values
        vec = np.array(embeddings[0])
        norm = np.linalg.norm(vec)
        print(f"Vector Norm: {norm:.4f}")

        if np.isnan(vec).any():
            print("❌ Vector contains NaNs")
            exit(1)

        print("✅ Live Embedding Verification Passed")
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as e:
        print(f"❌ Embedding test failed: {e}")
        if hasattr(e, "response") and e.response:
            print(f"Response: {e.response.text}")
        exit(1)


if __name__ == "__main__":
    test_live_embeddings()
