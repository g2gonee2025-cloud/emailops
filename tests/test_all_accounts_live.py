#!/usr/bin/env python3
"""
Make live Vertex AI API calls using all 6 credentials to verify they work
"""

import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from cortex.config.loader import get_config, reset_config
from cortex.llm.client import embed_texts
from cortex.llm.runtime import reset_vertex_init
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class Account:
    """Simple wrapper for account details to match previous test structure."""

    def __init__(self, path: Path):
        self.credentials_path = str(path)
        try:
            with path.open(encoding="utf-8") as f:
                data = json.load(f)
            self.project_id = data.get("project_id")
        except Exception as e:
            logger.warning(f"Failed to parse {path}: {e}")
            self.project_id = "unknown"
        self.account_group = 0  # Default/Dummy


def load_validated_accounts():
    """Load all validated accounts using the config loader."""
    config = get_config()
    paths = config.get_all_credential_files()
    return [Account(p) for p in paths]


def _init_vertex(project_id, credentials_path, location="us-central1"):
    """Helper to initialize vertex (shim for old function)."""
    import vertexai

    vertexai.init(project=project_id, location=location, credentials=credentials_path)


def make_live_api_call(account):
    """Make a live Vertex AI embedding API call with the given account."""
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Testing Account: {account.project_id}")
    logger.info(f"{'=' * 60}")

    try:
        # Reset any previous initialization
        reset_vertex_init()
        reset_config()

        # Set environment variables for this account
        os.environ["GCP_PROJECT"] = account.project_id
        os.environ["GOOGLE_CLOUD_PROJECT"] = account.project_id
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = account.credentials_path

        logger.info(f"DEBUG: GCP_PROJECT={os.environ['GCP_PROJECT']}")
        logger.info(
            f"DEBUG: GOOGLE_APPLICATION_CREDENTIALS={os.environ['GOOGLE_APPLICATION_CREDENTIALS']}"
        )
        cred_path = Path(account.credentials_path)
        if cred_path.exists():
            logger.info("DEBUG: Credential file exists")
        else:
            logger.info("DEBUG: Credential file DOES NOT EXIST")

        os.environ["EMBED_PROVIDER"] = "vertex"
        os.environ["VERTEX_EMBED_MODEL"] = "gemini-embedding-001"
        os.environ["GCP_REGION"] = "us-central1"

        # Initialize Vertex AI manually since we are bypassing the runtime's auto-init for specific accounts
        # Note: cortex.llm.runtime handles init internally, but here we want to force a specific account.
        # We rely on env vars being set above, and reset_vertex_init() clearing the state.
        # The runtime will pick up the new env vars when it re-initializes.

        # Make the actual API call
        start_time = time.time()
        test_texts = [
            f"Testing embedding API for project {account.project_id}",
            f"This is a live API call at {datetime.now().isoformat()}",
            "Verifying that all 6 accounts can make successful API calls",
        ]

        logger.info("Making live embedding API call...")
        embeddings = embed_texts(
            test_texts
        )  # provider arg is not needed if env var is set, or passed via config
        api_time = time.time() - start_time

        if embeddings is not None and embeddings.shape[0] == len(test_texts):
            logger.info(
                f"✓ SUCCESS: Got {embeddings.shape[0]} embeddings with dimension {embeddings.shape[1]}"
            )
            logger.info(f"✓ API call completed in {api_time:.2f} seconds")
            logger.info(
                f"✓ First embedding vector (first 10 values): {embeddings[0][:10].tolist()}"
            )
            return True, {
                "status": "success",
                "embeddings_shape": embeddings.shape,
                "api_time": api_time,
                "sample_values": embeddings[0][:10].tolist(),
            }
        else:
            logger.error("✗ FAILED: No embeddings returned")
            return False, {"status": "failed", "error": "No embeddings returned"}

    except Exception as e:
        logger.error(f"✗ FAILED: {e!s}")
        return False, {"status": "failed", "error": str(e)}


def main():
    """Run live API tests on all accounts"""
    logger.info("VERTEX AI LIVE API TEST - ALL 6 ACCOUNTS")
    logger.info("========================================\n")

    try:
        # Load all accounts
        accounts = load_validated_accounts()
        logger.info(f"Loaded {len(accounts)} validated accounts")

        # Test 1: Individual API calls
        individual_results = []
        for i, account in enumerate(accounts):
            logger.info(f"\n[TEST {i + 1}/6] Individual API call")
            success, result = make_live_api_call(account)
            individual_results.append(
                {
                    "account": account.project_id,
                    "credentials": account.credentials_path,
                    "account_group": account.account_group,
                    "success": success,
                    "details": result,
                }
            )

        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("LIVE API TEST SUMMARY")
        logger.info("=" * 60)

        successful = sum(1 for r in individual_results if r["success"])
        logger.info(f"\nIndividual API Calls: {successful}/{len(accounts)} successful")

        for i, result in enumerate(individual_results):
            status = "✓ SUCCESS" if result["success"] else "✗ FAILED"
            logger.info(f"\n{i + 1}. {result['account']} - {status}")
            if result["success"]:
                details = result["details"]
                logger.info(f"   - Embeddings shape: {details['embeddings_shape']}")
                logger.info(f"   - API response time: {details['api_time']:.2f}s")
            else:
                logger.info(f"   - Error: {result['details']['error']}")

        # Save results
        output_file = Path("live_api_test_results.json")
        with output_file.open("w") as f:
            json.dump(
                {
                    "timestamp": datetime.now().isoformat(),
                    "total_accounts": len(accounts),
                    "successful_calls": successful,
                    "individual_results": individual_results,
                },
                f,
                indent=2,
            )

        logger.info(f"\nDetailed results saved to: {output_file}")

        # Final verdict
        if successful == len(accounts):
            logger.info("\n✅ ALL 6 ACCOUNTS ARE WORKING PERFECTLY!")
            logger.info("All accounts can make successful Vertex AI API calls.")
        else:
            logger.info(f"\n⚠️  {successful}/{len(accounts)} accounts working")

        return 0 if successful == len(accounts) else 1

    except Exception as e:
        logger.error(f"Error during live API test: {e!s}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
