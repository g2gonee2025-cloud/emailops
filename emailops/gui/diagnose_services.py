"""
Diagnostic script to identify which services are failing and why.
"""

import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from emailops.core_config import get_config
from emailops.services import (
    AnalysisService,
    BatchService,
    ChatService,
    ChunkingService,
    ConfigService,
    EmailService,
    FileService,
    IndexingService,
    SearchService,
)

print("Starting service diagnostics...", flush=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def diagnose_services():
    """Diagnose which services are failing and why."""

    print("Loading configuration...", flush=True)
    # Get configuration
    config = get_config()
    config_path = Path.home() / ".emailops" / "config.json"
    export_root_path = (
        Path(config.core.export_root)
        if config.core.export_root
        else config_path.parent
    )

    if not export_root_path.exists():
        export_root_path.mkdir(parents=True, exist_ok=True)

    export_root = str(export_root_path)
    index_dirname = config.directories.index_dirname

    print(f"Export root: {export_root}", flush=True)
    print(f"Index dirname: {index_dirname}", flush=True)
    print(f"Config path: {config_path}", flush=True)

    results = {}

    # Test each service
    print("\n" + "="*60)
    print("SERVICE DIAGNOSTIC RESULTS")
    print("="*60 + "\n")

    # 1. SearchService
    try:
        search_service = SearchService(export_root, index_dirname=index_dirname)
        valid, msg = search_service.validate_index()
        results["search"] = valid
        print("✓ SearchService initialized")
        print(f"  - validate_index(): {valid}")
        if not valid:
            print(f"  - Reason: {msg}")
    except Exception as e:
        results["search"] = False
        print(f"✗ SearchService FAILED: {e}")

    # 2. EmailService
    try:
        EmailService(export_root)
        results["email"] = True
        print("✓ EmailService initialized (always ready)")
    except Exception as e:
        results["email"] = False
        print(f"✗ EmailService FAILED: {e}")

    # 3. ChunkingService
    try:
        chunking_service = ChunkingService(export_root)
        valid, msg = chunking_service.validate_export_root()
        results["chunking"] = valid
        print("✓ ChunkingService initialized")
        print(f"  - validate_export_root(): {valid}")
        if not valid:
            print(f"  - Reason: {msg}")
    except Exception as e:
        results["chunking"] = False
        print(f"✗ ChunkingService FAILED: {e}")

    # 4. IndexingService
    try:
        indexing_service = IndexingService(export_root, index_dirname=index_dirname)
        valid, msg = indexing_service.validate_index()
        results["indexing"] = valid
        print("✓ IndexingService initialized")
        print(f"  - validate_index(): {valid}")
        if not valid:
            print(f"  - Reason: {msg}")
    except Exception as e:
        results["indexing"] = False
        print(f"✗ IndexingService FAILED: {e}")

    # 5. AnalysisService
    try:
        AnalysisService(export_root)
        results["analysis"] = True
        print("✓ AnalysisService initialized (always ready)")
    except Exception as e:
        results["analysis"] = False
        print(f"✗ AnalysisService FAILED: {e}")

    # 6. BatchService
    try:
        BatchService(export_root)
        results["batch"] = True
        print("✓ BatchService initialized (always ready)")
    except Exception as e:
        results["batch"] = False
        print(f"✗ BatchService FAILED: {e}")

    # 7. ChatService
    try:
        chat_service = ChatService(export_root, index_dirname=index_dirname)
        valid = chat_service.index_dir.exists()
        results["chat"] = valid
        print("✓ ChatService initialized")
        print(f"  - index_dir.exists(): {valid}")
        print(f"  - index_dir: {chat_service.index_dir}")
    except Exception as e:
        results["chat"] = False
        print(f"✗ ChatService FAILED: {e}")

    # 8. FileService
    try:
        FileService(export_root)
        results["file"] = True
        print("✓ FileService initialized (always ready)")
    except Exception as e:
        results["file"] = False
        print(f"✗ FileService FAILED: {e}")

    # 9. ConfigService
    try:
        config_service = ConfigService(config_path)
        valid = config_service.current_config is not None
        results["config"] = valid
        print("✓ ConfigService initialized")
        print(f"  - current_config is not None: {valid}")
    except Exception as e:
        results["config"] = False
        print(f"✗ ConfigService FAILED: {e}")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    operational = sum(1 for v in results.values() if v)
    total = len(results)
    failed_services = [name for name, status in results.items() if not status]

    print(f"\nOperational: {operational}/{total}")

    if failed_services:
        print("\nFailed services:")
        for svc in failed_services:
            print(f"  - {svc}")
    else:
        print("\nAll services operational!")

    return results

if __name__ == "__main__":
    diagnose_services()
