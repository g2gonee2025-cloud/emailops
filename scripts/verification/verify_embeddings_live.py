import os
import sys
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
        response = requests.post(f"{BASE_URL}/v1/embeddings", json=payload, timeout=30)
        response.raise_for_status()
        duration = time.time() - start_time

        # vLLM returns OpenAI-compatible format
        data = response.json()
        if not isinstance(data, dict):
            raise ValueError("Unexpected response format: expected a JSON object.")
        embeddings = []
        if "data" in data and isinstance(data["data"], list):
            for idx, item in enumerate(data["data"]):
                if (
                    isinstance(item, dict)
                    and "embedding" in item
                    and isinstance(item["embedding"], list)
                ):
                    embeddings.append(item["embedding"])
                else:
                    print(
                        f"⚠️ Skipping item at index {idx} without a valid 'embedding' field"
                    )
        elif "embedding" in data and isinstance(data["embedding"], list):
            embeddings = [data["embedding"]]
        elif "embeddings" in data and isinstance(data["embeddings"], list):
            embeddings = data["embeddings"]

        if not embeddings:
            raise ValueError("No valid embeddings found in the response.")

        print(f"✅ Request successful in {duration:.2f}s")
        print(f"Received {len(embeddings)} embeddings")

        # Validate dimensions
        first_embedding = embeddings[0]
        if not isinstance(first_embedding, list):
            print(f"❌ Embedding is not a list: {type(first_embedding)}")
            sys.exit(1)
        dim = len(first_embedding)
        print(f"Vector Dimension: {dim}")

        if dim == EXPECTED_DIM:
            print(f"✅ Dimension matches expected {EXPECTED_DIM}")
        else:
            print(f"❌ Dimension mismatch! Expected {EXPECTED_DIM}, got {dim}")
            sys.exit(1)

        # Validate values
        try:
            vec = np.array(first_embedding, dtype=np.float64)
        except ValueError:
            print("❌ Could not convert embedding to a numeric numpy array.")
            sys.exit(1)

        norm = np.linalg.norm(vec)
        print(f"Vector Norm: {norm:.4f}")

        if np.isnan(vec).any():
            print("❌ Vector contains NaNs")
            sys.exit(1)

        print("✅ Live Embedding Verification Passed")
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as e:
        print(f"❌ Embedding test failed: {e}")
        if hasattr(e, "response") and e.response:
            print(f"Response: {e.response.text}")
        sys.exit(1)


if __name__ == "__main__":
    test_live_embeddings()
