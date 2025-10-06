from __future__ import annotations
import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

# Library-safe logging: no basicConfig at module level
logger = logging.getLogger(__name__)

# Custom exception class for LLM-related errors
class LLMError(Exception):
    """Custom exception for LLM-related errors."""
    pass

# Track initialization state to avoid redundant calls
_vertex_initialized = False
_validated_accounts: Optional[List[Dict[str, Any]]] = None

@dataclass
class VertexAccount:
    """Validated Vertex AI account configuration"""
    project_id: str
    credentials_path: str
    account_group: int = 0
    is_valid: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "project_id": self.project_id,
            "credentials_path": self.credentials_path,
            "account_group": self.account_group
        }

def load_validated_accounts(
    validated_file: str = "validated_accounts.json",
    default_accounts: Optional[List[Dict[str, str]]] = None
) -> List[VertexAccount]:
    """
    Load and validate GCP accounts from validated_accounts.json or defaults.
    
    Args:
        validated_file: Path to validated accounts JSON file
        default_accounts: Fallback accounts if file doesn't exist
        
    Returns:
        List of validated VertexAccount objects
    """
    global _validated_accounts
    
    if _validated_accounts is not None:
        return _validated_accounts
    
    accounts = []
    validated_path = Path(validated_file)
    
    # Try to load from validated_accounts.json
    if validated_path.exists():
        try:
            with open(validated_path, 'r') as f:
                data = json.load(f)
                account_list = data.get("accounts", [])
                
                logger.info(f"Loaded {len(account_list)} validated accounts from {validated_file}")
                
                for idx, acc in enumerate(account_list):
                    # Determine account group (0 = Account 1, 1 = Account 2)
                    # First 3 projects are Account 1, rest are Account 2
                    account_group = 0 if idx < 3 else 1
                    
                    accounts.append(VertexAccount(
                        project_id=acc['project_id'],
                        credentials_path=acc['credentials_path'],
                        account_group=account_group,
                        is_valid=True
                    ))
                    
        except (json.JSONDecodeError, IOError, KeyError) as e:
            logger.warning(f"Failed to load validated accounts: {e}")
    
    # Fallback to default accounts if provided
    if not accounts and default_accounts:
        logger.info("Using default accounts (no validated_accounts.json found)")
        for idx, acc in enumerate(default_accounts):
            account_group = 0 if idx < 3 else 1
            accounts.append(VertexAccount(
                project_id=acc['project_id'],
                credentials_path=acc['credentials_path'],
                account_group=account_group,
                is_valid=True
            ))
    
    # Validate credential files exist
    valid_accounts = []
    for acc in accounts:
        creds_path = Path(acc.credentials_path)
        if not creds_path.exists():
            logger.warning(f"Credentials file not found for {acc.project_id}: {acc.credentials_path}")
            acc.is_valid = False
        else:
            valid_accounts.append(acc)
    
    if not valid_accounts:
        raise LLMError(
            "No valid GCP accounts found. Please ensure:\n"
            "1. validated_accounts.json exists with valid credentials, OR\n"
            "2. Default accounts have valid credential files in secrets/ directory"
        )
    
    logger.info(f"Found {len(valid_accounts)} accounts with valid credentials")
    _validated_accounts = valid_accounts
    return valid_accounts

def get_worker_configs() -> List[VertexAccount]:
    """
    Get worker configurations based on validated accounts.
    Number of workers equals number of valid accounts.
    
    Returns:
        List of VertexAccount objects ready for parallel processing
    """
    accounts = load_validated_accounts()
    logger.info(f"Configured {len(accounts)} workers from validated accounts")
    return accounts

def save_validated_accounts(
    accounts: List[VertexAccount],
    output_file: str = "validated_accounts.json"
) -> None:
    """
    Save validated accounts to JSON file.
    
    Args:
        accounts: List of VertexAccount objects to save
        output_file: Output JSON file path
    """
    from datetime import datetime
    
    data = {
        "accounts": [acc.to_dict() for acc in accounts if acc.is_valid],
        "timestamp": datetime.now().isoformat(),
        "total_working": len([acc for acc in accounts if acc.is_valid])
    }
    
    output_path = Path(output_file)
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    logger.info(f"Saved {data['total_working']} validated accounts to {output_file}")

def _init_vertex(
    project_id: Optional[str] = None,
    credentials_path: Optional[str] = None,
    location: Optional[str] = None
) -> None:
    """
    Initialize Vertex AI SDK with proper credentials.
    
    Args:
        project_id: GCP project ID (uses env var if not provided)
        credentials_path: Path to service account JSON (uses env var if not provided)
        location: GCP region (default: "global" for Gemini models)
    """
    global _vertex_initialized
    
    if _vertex_initialized:
        return  # Already initialized
    
    import vertexai
    
    # Get project from args or environment
    project = project_id or os.getenv("GCP_PROJECT") or os.getenv("GOOGLE_CLOUD_PROJECT")
    location = location or os.getenv("GCP_REGION", "global")  # Changed to "global" for Gemini
    
    if not project:
        raise LLMError(
            "GCP project not specified. Set one of:\n"
            "- GCP_PROJECT environment variable\n"
            "- GOOGLE_CLOUD_PROJECT environment variable\n"
            "- Pass project_id parameter to _init_vertex()"
        )
    
    # Get credentials path from args or environment
    service_account_path = credentials_path or os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    
    try:
        if service_account_path:
            creds_path = Path(service_account_path)
            
            # Resolve relative paths
            if not creds_path.is_absolute():
                module_dir = Path(__file__).resolve().parent.parent
                creds_path = module_dir / service_account_path
            
            if creds_path.exists():
                from google.oauth2 import service_account
                credentials = service_account.Credentials.from_service_account_file(str(creds_path))
                vertexai.init(project=project, location=location, credentials=credentials)
                logger.info(f"? Vertex AI initialized with service account: {creds_path.name}")
            else:
                logger.warning(
                    f"GOOGLE_APPLICATION_CREDENTIALS set to '{service_account_path}' but file not found. "
                    f"Falling back to Application Default Credentials."
                )
                vertexai.init(project=project, location=location)
                logger.info("? Vertex AI initialized with Application Default Credentials")
        else:
            # Application Default Credentials (e.g., Workbench SA or gcloud ADC)
            vertexai.init(project=project, location=location)
            logger.info("? Vertex AI initialized with Application Default Credentials")
        
        _vertex_initialized = True
        
    except Exception as e:
        raise LLMError(f"Failed to initialize Vertex AI SDK: {e}") from e

def reset_vertex_init() -> None:
    """Reset Vertex AI initialization state (useful for testing or project switching)"""
    global _vertex_initialized
    _vertex_initialized = False
    logger.debug("Vertex AI initialization state reset")

def validate_account(account: VertexAccount) -> Tuple[bool, str]:
    """
    Validate a GCP account by checking credentials and API access.
    
    Args:
        account: VertexAccount to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check credentials file exists
    creds_path = Path(account.credentials_path)
    if not creds_path.exists():
        return False, f"Credentials file not found: {account.credentials_path}"
    
    # Try to initialize Vertex AI
    try:
        reset_vertex_init()
        _init_vertex(
            project_id=account.project_id,
            credentials_path=account.credentials_path
        )
        return True, "OK"
    except Exception as e:
        return False, str(e)

# Default accounts for fallback (same as vertex_indexer.py)
DEFAULT_ACCOUNTS = [
    {"project_id": "api-agent-470921", "credentials_path": "secrets/api-agent-470921-4e2065b2ecf9.json"},
    {"project_id": "apt-arcana-470409-i7", "credentials_path": "secrets/apt-arcana-470409-i7-ce42b76061bf.json"},
    {"project_id": "embed2-474114", "credentials_path": "secrets/embed2-474114-fca38b4d2068.json"},
    {"project_id": "crafty-airfoil-474021-s2", "credentials_path": "secrets/crafty-airfoil-474021-s2-34159960925b.json"},
    {"project_id": "my-project-31635v", "credentials_path": "secrets/my-project-31635v-8ec357ac35b2.json"},
    {"project_id": "semiotic-nexus-470620-f3", "credentials_path": "secrets/semiotic-nexus-470620-f3-3240cfaf6036.json"}
]
