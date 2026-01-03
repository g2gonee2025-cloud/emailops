import logging
import os
import sys
import json

import requests

# Set path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(SCRIPT_DIR, "..", "backend", "src"))


# Configure logging
logging.basicConfig(level=logging.INFO)

from cortex.config.loader import get_config
from cortex.llm.runtime import LLMRuntime


def mask(s):
    if not s or len(s) < 8:
        return "[REDACTED]"
    return s[:4] + "..." + s[-4:] + f" (len={len(s)})"

def main():
    logging.info("--- DEBUG START ---")
    # 1. Inspect _config
    try:
        cfg = get_config()
        val = getattr(cfg, "llm_api_key", "MISSING_ATTR")
        logging.info(f"_config.llm_api_key: '{mask(val)}'")
    except Exception:
        logging.exception("Error loading config")

    # 2. Inspect Env Vars
    logging.info(f"Env DO_LLM_API_KEY: {mask(os.getenv('DO_LLM_API_KEY'))}")
    logging.info(f"Env LLM_API_KEY: {mask(os.getenv('LLM_API_KEY'))}")

    # 3. Inspect LLMRuntime resolution
    try:
        runtime = LLMRuntime()
        if runtime.primary and runtime.primary.llm_client:
            client = runtime.primary.llm_client  # Triggers init
            logging.info(f"Resolved Client API Key: {mask(client.api_key)}")
            logging.info(f"Resolved Client Base URL: {client.base_url}")
        else:
            logging.error("LLMRuntime initialization failed: primary client is None.")
    except Exception:
        logging.exception("Error initializing runtime")

    # 4. Raw Request Test
    try:
        key = os.getenv("DO_LLM_API_KEY")
        if not key:
            logging.warning("DO_LLM_API_KEY is not set. Skipping raw request test.")
        else:
            url = "https://inference.do-ai.run/v1/chat/completions"
            headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
            data = {
                "model": "gpt-oss-120b",
                "messages": [{"role": "user", "content": "ping"}],
                "max_tokens": 1,
            }
            logging.info(f"Attempting raw POST to {url} with key {mask(key)}...")
            resp = requests.post(url, headers=headers, json=data, timeout=5)
            logging.info(f"Raw Response Status: {resp.status_code}")
            try:
                body = resp.json()
                if "error" in body:
                    logging.warning(f"Raw Response Body contains error: {body['error']}")
                else:
                    logging.info("Raw Response Body: [REDACTED]")
            except json.JSONDecodeError:
                logging.info("Raw Response Body: [REDACTED - Not JSON]")
    except Exception:
        logging.exception("Raw request failed")

    logging.info("--- DEBUG END ---")

if __name__ == "__main__":
    main()
