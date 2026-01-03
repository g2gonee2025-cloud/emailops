import os
import sys
import time
from typing import List

import requests

# --- Configuration ---
# Read API key and base URL from environment. Do NOT hardcode secrets.
# Supported env vars:
#   - DO_API_KEY or OPENAI_API_KEY
#   - DO_BASE_URL (defaults to https://inference.do-ai.run/v1)
api_key = os.getenv("DO_API_KEY") or os.getenv("OPENAI_API_KEY")
BASE_URL = (os.getenv("DO_BASE_URL") or "https://inference.do-ai.run/v1").rstrip("/")

if not api_key:
    print(
        "Missing API key. Set DO_API_KEY or OPENAI_API_KEY in your environment and re-run."
    )
    sys.exit(1)

headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

# Desired models (will be intersected with available ones from /models)
DESIRED_MODELS = [
    "openai-gpt-oss-120b",
]


def fetch_available_models() -> List[str]:
    """Return a list of model IDs available on the endpoint."""
    url = f"{BASE_URL}/models"
    try:
        resp = requests.get(url, headers=headers, timeout=30)
        if resp.status_code != 200:
            # Provide actionable error messages
            if resp.status_code == 401:
                raise RuntimeError("401 Unauthorized: Invalid/expired API key")
            raise RuntimeError(
                f"{resp.status_code} Error fetching /models: {resp.text}"
            )
        payload = resp.json()
        # OpenAI-compatible: {"data": [{"id": "model-id", ...}, ...]}
        items = payload.get("data")
        # Fallback for other response formats
        if items is None:
            items = payload.get("models") or []
        model_ids = []
        for it in items:
            if isinstance(it, dict) and "id" in it:
                model_ids.append(it["id"])
            elif isinstance(it, str):
                model_ids.append(it)
        return model_ids
    except requests.RequestException as e:
        raise RuntimeError(f"Network error fetching /models: {e}") from e


def test_model(model_name):
    """Test a specific model and return response details"""
    url = f"{BASE_URL}/chat/completions"

    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": "Please respond with just the text: 'This is a test of model identification'",
            }
        ],
        "max_tokens": 64,
        "temperature": 0.1,
    }

    try:
        start_time = time.time()
        response = requests.post(url, json=payload, headers=headers, timeout=120)
        response_time = time.time() - start_time

        if response.status_code == 200:
            data = response.json()
            return {
                "model_requested": model_name,
                "model_returned": data.get("model", "Unknown"),
                "response": (
                    data.get("choices")[0].get("message", {}).get("content", "")
                    if data.get("choices")
                    else ""
                ),
                "status_code": response.status_code,
                "response_time": response_time,
            }
        else:
            # Friendly classification for common errors
            if response.status_code == 401:
                err = "HTTP 401 Unauthorized: Check your API key"
            elif response.status_code == 404:
                err = "HTTP 404: model not found on this endpoint"
            elif response.status_code == 429:
                err = "HTTP 429: rate limited"
            else:
                err = f"HTTP {response.status_code}: An unexpected error occurred"
            return {
                "model_requested": model_name,
                "error": err,
                "status_code": response.status_code,
                "response_time": response_time,
            }

    except requests.RequestException as e:
        return {
            "model_requested": model_name,
            "error": f"Network error: {e}",
            "status_code": None,
            "response_time": None,
        }
    except ValueError as e:
        return {
            "model_requested": model_name,
            "error": f"JSON decode error: {e}",
            "status_code": None,
            "response_time": None,
        }


def main():
    """Main function to discover and test models."""
    # Discover models and test
    print("Discovering available models...")
    print("=" * 60)

    try:
        available = fetch_available_models()
    except Exception as e:
        print(f"Failed to fetch /models: {e}")
        sys.exit(1)

    # Allow override via TEST_MODELS env (comma-separated)
    override = os.getenv("TEST_MODELS")
    if override:
        models = [m.strip() for m in override.split(",") if m.strip()]
    else:
        # Intersect desired list with available; if none, test all available
        desired_set = set(DESIRED_MODELS)
        models = [m for m in available if m in desired_set]
        if not models:
            models = available

    print(f"Total models available: {len(available)}")
    if available:
        preview = ", ".join(available[:10]) + ("..." if len(available) > 10 else "")
        print(f"Sample: {preview}")

    print("\nTesting models...")
    print("-" * 60)

    results = []
    for model in models:
        print(f"Testing {model}...")
        result = test_model(model)
        results.append(result)

        if "error" in result:
            print(f"❌ Error: {result['error']}")
        else:
            print(f"✅ Model returned: {result['model_returned']}")
            print(f"📝 Response: {result['response']}")
            print(f"⏱️  Response time: {result['response_time']:.2f}s")
        print("-" * 40)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY:")
    print("=" * 60)

    print(f"{'Model Requested':<25} {'Model Returned':<25} {'Status':<10} {'Time':<8}")
    print("-" * 70)

    for result in results:
        if "error" in result:
            status = "❌ ERROR"
            model_returned = "N/A"
            response_time = "N/A"
        else:
            if result["model_requested"] == result["model_returned"]:
                status = "✅ MATCH"
            else:
                status = "⚠️  MISMATCH"
            model_returned = result["model_returned"]
            response_time = f"{result['response_time']:.2f}s"

        print(
            f"{result['model_requested']:<25} {model_returned:<25} {status:<10} {response_time:<8}"
        )

    # Detailed analysis
    print("\n" + "=" * 60)
    print("DETAILED ANALYSIS:")
    print("=" * 60)

    for result in results:
        if "error" not in result:
            if result["model_requested"] == result["model_returned"]:
                print(f"✓ {result['model_requested']}: Model correctly identified")
            else:
                print(
                    f"✗ {result['model_requested']}: API returned '{result['model_returned']}' instead"
                )
        else:
            print(f"✗ {result['model_requested']}: Failed to test - {result['error']}")


if __name__ == "__main__":
    main()
