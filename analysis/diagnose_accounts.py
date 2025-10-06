#!/usr/bin/env python3
"""
Diagnostic script to test each Vertex AI account individually
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import the required modules
from emailops.env_utils import load_validated_accounts, _init_vertex, reset_vertex_init
from emailops.llm_client import embed_texts

def test_account(account):
    """Test a single account by attempting to initialize Vertex AI and perform a test embedding"""
    logger.info(f"\n{'='*60}")
    logger.info(f"Testing Account: {account.project_id}")
    logger.info(f"Credentials: {account.credentials_path}")
    logger.info(f"Account Group: {account.account_group + 1}")
    logger.info(f"{'='*60}")
    
    # Check if credentials file exists
    creds_path = Path(account.credentials_path)
    if not creds_path.exists():
        logger.error(f"❌ FAILED: Credentials file not found: {account.credentials_path}")
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
        os.environ["VERTEX_EMBED_MODEL"] = os.getenv("VERTEX_EMBED_MODEL", "gemini-embedding-001")
        os.environ["GCP_REGION"] = "global"
        
        # Initialize Vertex AI
        _init_vertex(
            project_id=account.project_id,
            credentials_path=account.credentials_path,
            location="global"
        )
        logger.info("✓ Vertex AI initialized successfully")
        
    except Exception as e:
        logger.error(f"❌ FAILED to initialize Vertex AI: {str(e)}")
        return False, f"Vertex AI init failed: {str(e)}"
    
    # Try to perform a test embedding
    try:
        test_texts = ["This is a test embedding for diagnostics"]
        logger.info("Attempting test embedding...")
        
        embeddings = embed_texts(test_texts, provider="vertex")
        
        if embeddings and len(embeddings) > 0:
            logger.info(f"✓ Test embedding successful! Shape: {embeddings.shape}")
            return True, "All tests passed"
        else:
            logger.error("❌ FAILED: No embeddings returned")
            return False, "No embeddings returned"
            
    except Exception as e:
        logger.error(f"❌ FAILED to create embedding: {str(e)}")
        return False, f"Embedding failed: {str(e)}"

def main():
    """Test all validated accounts"""
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
            results.append({
                "project_id": account.project_id,
                "credentials_path": account.credentials_path,
                "account_group": account.account_group,
                "success": success,
                "message": message
            })
            
            if success:
                working_accounts.append(account)
            else:
                failed_accounts.append(account)
                
            # Small delay between tests to avoid rate limiting
            if i < len(accounts) - 1:
                import time
                time.sleep(2)
        
        # Summary
        logger.info(f"\n{'='*60}")
        logger.info("SUMMARY")
        logger.info(f"{'='*60}")
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
            logger.info(f"  Group {group + 1}: {stats['working']}/{stats['total']} working")
        
        # List failed accounts
        if failed_accounts:
            logger.info("\nFailed Accounts:")
            for acc in failed_accounts:
                result = next(r for r in results if r["project_id"] == acc.project_id)
                logger.info(f"  - {acc.project_id}: {result['message']}")
        
        # Save diagnostic results
        output_file = "account_diagnostics.json"
        with open(output_file, 'w') as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "total_accounts": len(accounts),
                "working_accounts": len(working_accounts),
                "failed_accounts": len(failed_accounts),
                "results": results
            }, f, indent=2)
        
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
        logger.error(f"Error during diagnostics: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
