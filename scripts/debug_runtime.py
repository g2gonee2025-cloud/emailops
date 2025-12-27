import logging
import os
import sys

import requests

# Set path
sys.path.append("backend/src")

# Configure logging
logging.basicConfig(level=logging.ERROR)

from cortex.config.loader import get_config
from cortex.llm.runtime import LLMRuntime


def mask(s):
    if not s:
        return str(s)
    if len(s) < 8:
        return s
    return s[:4] + "..." + s[-4:] + f" (len={len(s)})"


print("--- DEBUG START ---")
# 1. Inspect _config
try:
    cfg = get_config()
    val = getattr(cfg, "llm_api_key", "MISSING_ATTR")
    print(f"_config.llm_api_key: '{val}'")
except Exception as e:
    print(f"Error loading config: {e}")

# 2. Inspect Env Vars
print(f"Env DO_LLM_API_KEY: {mask(os.getenv('DO_LLM_API_KEY'))}")
print(f"Env LLM_API_KEY: {mask(os.getenv('LLM_API_KEY'))}")

# 3. Inspect LLMRuntime resolution
try:
    runtime = LLMRuntime()
    client = runtime.primary.llm_client  # Triggers init
    print(f"Resolved Client API Key: {mask(client.api_key)}")
    print(f"Resolved Client Base URL: {client.base_url}")
except Exception as e:
    print(f"Error initializing runtime: {e}")

# 4. Raw Request Test
try:
    key = os.getenv("DO_LLM_API_KEY")
    url = "https://inference.do-ai.run/v1/chat/completions"
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    data = {
        "model": "gpt-oss-120b",
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 1,
    }
    print(f"Attempting raw POST to {url} with key {mask(key)}...")
    resp = requests.post(url, headers=headers, json=data, timeout=5)
    print(f"Raw Response Status: {resp.status_code}")
    print(f"Raw Response Body: {resp.text}")
except Exception as e:
    print(f"Raw request failed: {e}")

print("--- DEBUG END ---")
