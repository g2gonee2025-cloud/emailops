import os
import time

import numpy as np
import requests

# Configuration
BASE_URL = os.getenv("DO_LLM_BASE_URL", "http://localhost:8081")
MODEL_ID = "tencent/KaLM-Embedding-Gemma3-12B-2511"
EXPECTED_DIM = 3840


def test_live_embeddings():
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
        embeddings = [item["embedding"] for item in data["data"]]

        print(f"✅ Request successful in {duration:.2f}s")
        print(f"Received {len(embeddings)} embeddings")

        # Validate dimensions
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

    except Exception as e:
        print(f"❌ Embedding test failed: {e}")
        if hasattr(e, "response") and e.response:
            print(f"Response: {e.response.text}")
        exit(1)


if __name__ == "__main__":
    test_live_embeddings()
