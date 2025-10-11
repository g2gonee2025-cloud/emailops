#!/usr/bin/env python3
"""
Unified diagnostic module for testing Vertex AI accounts and verifying index integrity.
Consolidates functionality from diagnose_accounts.py and verify_index_alignment.py.
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np

from diagnostics.utils import (
    format_timestamp,
    get_index_path,
    save_json_report,
    setup_logging,
)
from emailops.llm_runtime import (_init_vertex, load_validated_accounts, reset_vertex_init, VertexAccount)
from emailops.llm_client import embed_texts

# Setup logging
logger = setup_logging()

# Required fields for index validation
REQUIRED_FIELDS = [
    "id",
    "path",
    "conv_id",
    "doc_type",
    "subject",
    "snippet",
    "modified_time",
]

VALID_DOCTYPES = {"conversation", "attachment"}


def test_account(account) -> tuple[bool, str]:
    """
    Test a single Vertex AI account by attempting to initialize and perform a test embedding.

    Args:
        account: VertexAccount object with project_id, credentials_path, account_group

    Returns:
        Tuple of (success: bool, message: str)
    """
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Testing Account: {account.project_id}")
    logger.info(f"Credentials: {account.credentials_path}")
    logger.info(f"Account Group: {account.account_group + 1}")
    logger.info(f"{'=' * 60}")

    # Check if credentials file exists
    creds_path = Path(account.credentials_path)
    if not creds_path.exists():
        logger.error(
            f"❌ FAILED: Credentials file not found: {account.credentials_path}"
        )
        return False, "Credentials file not found"

    logger.info("✓ Credentials file exists")

    # Try to initialize Vertex AI
    try:
        # Reset any previous initialization
        reset_vertex_init()

        # Set environment variables
        os.environ["GCP_PROJECT"] = account.project_id
        os.environ["GOOGLE_CLOUD_PROJECT"] = account.project_id
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = account.credentials_path
        os.environ["EMBED_PROVIDER"] = "vertex"
        os.environ["VERTEX_EMBED_MODEL"] = os.getenv(
            "VERTEX_EMBED_MODEL", "gemini-embedding-001"
        )
        os.environ["GCP_REGION"] = "global"

        # Initialize Vertex AI
        _init_vertex(
            project_id=account.project_id,
            credentials_path=account.credentials_path,
            location="global",
        )
        logger.info("✓ Vertex AI initialized successfully")

    except Exception as e:
        logger.error(f"❌ FAILED to initialize Vertex AI: {e!s}")
        return False, f"Vertex AI init failed: {e!s}"

    # Try to perform a test embedding
    try:
        test_texts = ["This is a test embedding for diagnostics"]
        logger.info("Attempting test embedding...")

        embeddings = embed_texts(test_texts, provider="vertex")

        if embeddings is not None and len(embeddings) > 0:
            logger.info(f"✓ Test embedding successful! Shape: {embeddings.shape}")
            return True, "All tests passed"
        else:
            logger.error("❌ FAILED: No embeddings returned")
            return False, "No embeddings returned"

    except Exception as e:
        logger.error(f"❌ FAILED to create embedding: {e!s}")
        return False, f"Embedding failed: {e!s}"


def diagnose_all_accounts() -> int:
    """
    Main entry point for account diagnostics. Tests all validated accounts.

    Returns:
        Exit code (0 for success, 1 for failures)
    """
    logger.info("VERTEX AI ACCOUNT DIAGNOSTICS")
    logger.info("==============================\n")

    try:
        # Load validated accounts
        accounts = load_validated_accounts()
        logger.info(f"Found {len(accounts)} validated accounts\n")

        # Test each account
        results = []
        working_accounts = []
        failed_accounts = []

        for i, account in enumerate(accounts):
            success, message = test_account(account)
            results.append(
                {
                    "project_id": account.project_id,
                    "credentials_path": account.credentials_path,
                    "account_group": account.account_group,
                    "success": success,
                    "message": message,
                }
            )

            if success:
                working_accounts.append(account)
            else:
                failed_accounts.append(account)

            # Small delay between tests to avoid rate limiting
            if i < len(accounts) - 1:
                time.sleep(2)

        # Summary
        logger.info(f"\n{'=' * 60}")
        logger.info("SUMMARY")
        logger.info(f"{'=' * 60}")
        logger.info(f"Total accounts tested: {len(accounts)}")
        logger.info(f"✓ Working accounts: {len(working_accounts)}")
        logger.info(f"❌ Failed accounts: {len(failed_accounts)}")

        # Group by account group
        group_stats = {}
        for acc in accounts:
            group = acc.account_group
            if group not in group_stats:
                group_stats[group] = {"total": 0, "working": 0}
            group_stats[group]["total"] += 1

        for acc in working_accounts:
            group_stats[acc.account_group]["working"] += 1

        logger.info("\nBy Account Group:")
        for group, stats in sorted(group_stats.items()):
            logger.info(
                f"  Group {group + 1}: {stats['working']}/{stats['total']} working"
            )

        # List failed accounts
        if failed_accounts:
            logger.info("\nFailed Accounts:")
            for acc in failed_accounts:
                result = next(r for r in results if r["project_id"] == acc.project_id)
                logger.info(f"  - {acc.project_id}: {result['message']}")

        # Save diagnostic results
        output_file = "account_diagnostics.json"
        save_json_report(
            {
                "timestamp": datetime.now().isoformat(),
                "total_accounts": len(accounts),
                "working_accounts": len(working_accounts),
                "failed_accounts": len(failed_accounts),
                "results": results,
            },
            output_file,
        )

        logger.info(f"\nDiagnostic results saved to: {output_file}")

        # Final recommendation
        if len(working_accounts) < len(accounts):
            logger.info("\n⚠️  RECOMMENDATION:")
            logger.info("Some accounts are failing. You may need to:")
            logger.info("1. Check if the Vertex AI API is enabled for failed projects")
            logger.info("2. Verify the service account permissions")
            logger.info("3. Check project quotas and billing status")
            logger.info("4. Run: python enable_vertex_apis.py")

        return 0 if len(working_accounts) == len(accounts) else 1

    except Exception as e:
        logger.error(f"Error during diagnostics: {e!s}")
        return 1


def verify_index_alignment(root: str) -> None:
    """
    Verify index integrity and alignment between mapping.json, embeddings.npy, and meta.json.

    Args:
        root: Root directory path containing the index

    Raises:
        SystemExit: Exits with code 2 if validation fails
    """
    ix = get_index_path(root)
    mp = ix / "mapping.json"
    ep = ix / "embeddings.npy"
    meta = ix / "meta.json"

    def fail(msg: str) -> None:
        """Print error message and exit."""
        logger.error(f"❌ {msg}")
        sys.exit(2)

    def ok(msg: str) -> None:
        """Print success message."""
        logger.info(f"✅ {msg}")

    # Check required files exist
    if not ix.exists():
        fail(f"Index dir missing: {ix}")
    if not mp.exists():
        fail(f"mapping.json missing at {mp}")
    if not ep.exists():
        fail(f"embeddings.npy missing at {ep}")

    # Load mapping
    try:
        mapping = json.loads(mp.read_text(encoding="utf-8"))
        if not isinstance(mapping, list) or not mapping:
            fail("mapping.json is empty or not a list")
    except Exception as e:
        fail(f"Failed to read mapping.json: {e}")
    ok(f"mapping.json loaded with {len(mapping)} rows")

    # Load embeddings
    try:
        embs = np.load(ep, mmap_mode="r")
        if embs.ndim != 2 or embs.shape[0] <= 0 or embs.shape[1] <= 0:
            fail(f"Invalid embeddings shape: {getattr(embs, 'shape', None)}")
    except Exception as e:
        fail(f"Failed to read embeddings.npy: {e}")
    ok(f"embeddings.npy shape OK: {embs.shape}")

    # Count alignment
    if embs.shape[0] != len(mapping):
        fail(f"Row mismatch: embeddings={embs.shape[0]} mapping={len(mapping)}")
    ok("Row counts aligned")

    # Field checks
    ids = set()
    for i, m in enumerate(mapping):
        for k in REQUIRED_FIELDS:
            if k not in m:
                fail(
                    f"Missing field '{k}' at mapping row {i} (id={m.get('id', '<no-id>')})"
                )
        did = str(m["id"]).strip()
        if did in ids:
            fail(f"Duplicate id found: {did}")
        ids.add(did)

        if m.get("doc_type") not in VALID_DOCTYPES:
            fail(f"Invalid doc_type at id={did}: {m.get('doc_type')}")

        if not isinstance(m.get("snippet", ""), str):
            fail(f"snippet must be string at id={did}")

        # Optional sanity
        if "chunk_index" in m and not isinstance(m["chunk_index"], int):
            fail(f"chunk_index must be int when present (id={did})")

    ok("Mapping fields, id uniqueness, doc_type values are valid")

    # Meta checks (optional)
    if meta.exists():
        try:
            md = json.loads(meta.read_text(encoding="utf-8"))
            if "actual_dimensions" in md:
                if int(md["actual_dimensions"]) != int(embs.shape[1]):
                    fail(
                        f"meta.json actual_dimensions={md['actual_dimensions']} != embeddings dim={embs.shape[1]}"
                    )
        except Exception as e:
            fail(f"Failed to read meta.json: {e}")
        ok("meta.json present and dimension matches")

    print("\n🎉 All alignment checks passed.")


def check_index_consistency(root: Path) -> dict[str, Any]:
    """
    Perform detailed consistency checks on the index and return results.

    Args:
        root: Path to root directory containing the index

    Returns:
        Dictionary with check results and recommendations
    """
    ix = get_index_path(str(root))

    results = {
        "timestamp": format_timestamp(datetime.now()),
        "index_path": str(ix),
        "checks": {},
        "errors": [],
        "warnings": [],
        "recommendations": [],
    }

    # Check if index directory exists
    if not ix.exists():
        results["errors"].append(f"Index directory not found: {ix}")
        results["checks"]["index_exists"] = False
        return results

    results["checks"]["index_exists"] = True

    # Check for required files
    mp = ix / "mapping.json"
    ep = ix / "embeddings.npy"
    meta = ix / "meta.json"

    results["checks"]["mapping_exists"] = mp.exists()
    results["checks"]["embeddings_exists"] = ep.exists()
    results["checks"]["meta_exists"] = meta.exists()

    if not mp.exists():
        results["errors"].append("mapping.json not found")
    if not ep.exists():
        results["errors"].append("embeddings.npy not found")

    # If critical files missing, return early
    if not (mp.exists() and ep.exists()):
        results["recommendations"].append(
            "Rebuild the index using the indexing process"
        )
        return results

    # Load and validate mapping
    try:
        mapping = json.loads(mp.read_text(encoding="utf-8"))
        results["checks"]["mapping_valid"] = True
        results["checks"]["mapping_count"] = len(mapping)
    except Exception as e:
        results["errors"].append(f"Failed to load mapping.json: {e}")
        results["checks"]["mapping_valid"] = False
        return results

    # Load and validate embeddings
    try:
        embs = np.load(ep, mmap_mode="r")
        results["checks"]["embeddings_valid"] = True
        results["checks"]["embeddings_shape"] = embs.shape
    except Exception as e:
        results["errors"].append(f"Failed to load embeddings.npy: {e}")
        results["checks"]["embeddings_valid"] = False
        return results

    # Check alignment
    if embs.shape[0] == len(mapping):
        results["checks"]["counts_aligned"] = True
    else:
        results["checks"]["counts_aligned"] = False
        results["errors"].append(
            f"Row count mismatch: embeddings={embs.shape[0]}, mapping={len(mapping)}"
        )
        results["recommendations"].append("Rebuild the index to fix alignment issues")

    # Check for duplicate IDs
    ids = [str(m.get("id", "")).strip() for m in mapping]
    unique_ids = set(ids)
    if len(ids) != len(unique_ids):
        results["checks"]["ids_unique"] = False
        results["errors"].append(f"Found {len(ids) - len(unique_ids)} duplicate IDs")
        results["recommendations"].append("Remove duplicate entries from mapping.json")
    else:
        results["checks"]["ids_unique"] = True

    # Check required fields
    missing_fields = []
    for i, m in enumerate(mapping[:100]):  # Sample first 100
        for field in REQUIRED_FIELDS:
            if field not in m:
                missing_fields.append((i, field))
                break

    if missing_fields:
        results["checks"]["required_fields"] = False
        results["warnings"].append(
            f"Missing required fields in {len(missing_fields)} entries (sampled first 100)"
        )
    else:
        results["checks"]["required_fields"] = True

    # Validate meta.json if exists
    if meta.exists():
        try:
            md = json.loads(meta.read_text(encoding="utf-8"))
            if "actual_dimensions" in md:
                if int(md["actual_dimensions"]) == int(embs.shape[1]):
                    results["checks"]["meta_dimensions_match"] = True
                else:
                    results["checks"]["meta_dimensions_match"] = False
                    results["warnings"].append(
                        f"Dimension mismatch in meta.json: {md['actual_dimensions']} vs {embs.shape[1]}"
                    )
        except Exception as e:
            results["warnings"].append(f"Could not validate meta.json: {e}")

    # Generate overall status
    if not results["errors"]:
        results["status"] = "HEALTHY"
    elif results["checks"].get("counts_aligned", False):
        results["status"] = "WARNING"
    else:
        results["status"] = "CRITICAL"

    return results


if __name__ == "__main__":
    # Support both account diagnostics and index verification
    if len(sys.argv) > 1:
        if sys.argv[1] == "--accounts":
            sys.exit(diagnose_all_accounts())
        else:
            verify_index_alignment(sys.argv[1])
    else:
        # Default to account diagnostics
        sys.exit(diagnose_all_accounts())
