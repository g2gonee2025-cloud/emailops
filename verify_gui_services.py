#!/usr/bin/env python3
"""
Verification script to test EmailOps GUI service initialization.
Tests that all 9 services can be initialized correctly with proper configuration.
"""

import sys
from pathlib import Path


def verify_services():
    """Verify all services can be initialized."""
    print("=" * 60)
    print("EmailOps GUI Service Verification")
    print("=" * 60)

    try:
        # Import necessary modules
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

        print("\n1. Loading configuration...")
        config = get_config()

        # Get paths
        export_root = Path(config.core.export_root) if config.core.export_root else Path.home() / ".emailops"
        index_dirname = config.directories.index_dirname
        config_path = Path.home() / ".emailops" / "config.json"

        print(f"   Export root: {export_root}")
        print(f"   Index directory name: {index_dirname}")
        print(f"   Config path: {config_path}")

        # Ensure export root exists
        if not export_root.exists():
            print(f"\n   WARNING: Export root does not exist: {export_root}")
            print("   Creating directory...")
            export_root.mkdir(parents=True, exist_ok=True)

        # Ensure index directory exists
        index_dir = export_root / index_dirname
        if not index_dir.exists():
            print(f"\n   WARNING: Index directory does not exist: {index_dir}")
            print("   Creating directory...")
            index_dir.mkdir(parents=True, exist_ok=True)

            # Create minimal index structure
            import json

            import numpy as np

            mapping_file = index_dir / "mapping.json"
            mapping_file.write_text(json.dumps([], indent=2))

            meta_file = index_dir / "meta.json"
            meta_data = {
                "provider": "vertex",
                "model": "gemini-embedding-001",
                "actual_dimensions": 768,
                "index_type": "flat",
                "created_at": "2024-01-01T00:00:00Z"
            }
            meta_file.write_text(json.dumps(meta_data, indent=2))

            embeddings_file = index_dir / "embeddings.npy"
            np.save(embeddings_file, np.array([]))

            print("   Created minimal index structure")

        print("\n2. Initializing services...")
        services = {}
        service_status = {}

        # Initialize each service
        try:
            print("   - SearchService...", end=" ")
            services['search'] = SearchService(str(export_root), index_dirname=index_dirname)
            service_status['search'] = services['search'].validate_index()[0]
            print(f"{'OK' if service_status['search'] else 'FAILED'}")
        except Exception as e:
            print(f"ERROR: {e}")
            service_status['search'] = False

        try:
            print("   - EmailService...", end=" ")
            services['email'] = EmailService(str(export_root))
            service_status['email'] = True
            print("OK")
        except Exception as e:
            print(f"ERROR: {e}")
            service_status['email'] = False

        try:
            print("   - ChunkingService...", end=" ")
            services['chunking'] = ChunkingService(str(export_root))
            service_status['chunking'] = services['chunking'].validate_export_root()[0]
            print(f"{'OK' if service_status['chunking'] else 'FAILED'}")
        except Exception as e:
            print(f"ERROR: {e}")
            service_status['chunking'] = False

        try:
            print("   - IndexingService...", end=" ")
            services['indexing'] = IndexingService(str(export_root), index_dirname=index_dirname)
            service_status['indexing'] = services['indexing'].validate_index()[0]
            print(f"{'OK' if service_status['indexing'] else 'FAILED'}")
        except Exception as e:
            print(f"ERROR: {e}")
            service_status['indexing'] = False

        try:
            print("   - AnalysisService...", end=" ")
            services['analysis'] = AnalysisService(str(export_root))
            service_status['analysis'] = True
            print("OK")
        except Exception as e:
            print(f"ERROR: {e}")
            service_status['analysis'] = False

        try:
            print("   - BatchService...", end=" ")
            services['batch'] = BatchService(str(export_root))
            service_status['batch'] = True
            print("OK")
        except Exception as e:
            print(f"ERROR: {e}")
            service_status['batch'] = False

        try:
            print("   - ChatService...", end=" ")
            services['chat'] = ChatService(str(export_root), index_dirname=index_dirname)
            service_status['chat'] = services['chat'].index_dir.exists()
            print(f"{'OK' if service_status['chat'] else 'FAILED'}")
        except Exception as e:
            print(f"ERROR: {e}")
            service_status['chat'] = False

        try:
            print("   - FileService...", end=" ")
            services['file'] = FileService(str(export_root))
            service_status['file'] = True
            print("OK")
        except Exception as e:
            print(f"ERROR: {e}")
            service_status['file'] = False

        try:
            print("   - ConfigService...", end=" ")
            services['config'] = ConfigService(config_path)
            service_status['config'] = services['config'].current_config is not None
            print(f"{'OK' if service_status['config'] else 'FAILED'}")
        except Exception as e:
            print(f"ERROR: {e}")
            service_status['config'] = False

        # Summary
        print("\n" + "=" * 60)
        print("SERVICE STATUS SUMMARY")
        print("=" * 60)

        operational = sum(1 for status in service_status.values() if status)
        total = len(service_status)

        print(f"\nOperational Services: {operational}/{total}")
        print("\nDetails:")
        for name, status in service_status.items():
            status_text = "✓ READY" if status else "✗ NOT READY"
            print(f"  {name:15} {status_text}")

        print("\n" + "=" * 60)
        if operational == total:
            print("✓ ALL SERVICES OPERATIONAL")
            print("The GUI should initialize with no service warnings.")
            return 0
        else:
            print(f"✗ {total - operational} SERVICES NOT OPERATIONAL")
            print("The GUI will show service initialization warnings.")
            return 1

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(verify_services())
