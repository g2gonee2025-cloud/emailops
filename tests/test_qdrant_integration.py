"""
Test script for Qdrant integration with EmailOps
This script tests the Qdrant connection and basic operations
"""

import sys
from pathlib import Path

import numpy as np

# Add emailops to path
sys.path.insert(0, str(Path(__file__).parent))

from setup.qdrant_client import QdrantVectorStore, test_qdrant_connection


def test_qdrant_operations():
    """Test Qdrant basic operations"""
    print("=" * 60)
    print("Testing Qdrant Integration for EmailOps")
    print("=" * 60)

    # Test connection
    print("\n1. Testing connection...")
    if not test_qdrant_connection("config/qdrant_config.yaml"):
        print("Failed to connect to Qdrant. Please ensure Qdrant is running.")
        return False

    try:
        # Initialize client
        print("\n2. Initializing Qdrant client...")
        store = QdrantVectorStore("config/qdrant_config.yaml")

        # Get collection info
        print("\n3. Getting collection info...")
        info = store.get_collection_info()

        if not info:
            print("   Collection does not exist. Creating...")
            store.create_collection(recreate=False)
            info = store.get_collection_info()

        print(f"   Collection: {info.get('name', 'N/A')}")
        print(f"   Vectors count: {info.get('vectors_count', 0)}")
        print(f"   Points count: {info.get('points_count', 0)}")
        print(f"   Status: {info.get('status', 'unknown')}")

        # Test insertion
        print("\n4. Testing vector insertion...")

        # Create sample embeddings (3 vectors of dimension 768)
        embeddings = np.random.randn(3, 768).astype(np.float32)
        # Normalize them (cosine similarity works better with normalized vectors)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        # Create sample metadata
        metadata = [
            {
                "id": 1001,
                "doc_id": "test_doc_1",
                "text": "This is a test document for Qdrant integration",
                "source": "test",
                "timestamp": "2025-01-10T12:00:00Z"
            },
            {
                "id": 1002,
                "doc_id": "test_doc_2",
                "text": "Another test document with different content",
                "source": "test",
                "timestamp": "2025-01-10T12:01:00Z"
            },
            {
                "id": 1003,
                "doc_id": "test_doc_3",
                "text": "Third test document for vector search testing",
                "source": "test",
                "timestamp": "2025-01-10T12:02:00Z"
            }
        ]

        # Insert vectors
        count = store.upsert_embeddings(embeddings, metadata)
        print(f"   Inserted {count} vectors successfully")

        # Test search
        print("\n5. Testing vector search...")

        # Create a query vector (similar to first document)
        query_vector = embeddings[0] + np.random.randn(768) * 0.1
        query_vector = query_vector / np.linalg.norm(query_vector)

        # Search for similar vectors
        results = store.search(query_vector, limit=3)

        print(f"   Found {len(results)} similar vectors:")
        for i, (doc_id, score, payload) in enumerate(results):
            print(f"   {i+1}. ID: {doc_id}, Score: {score:.4f}")
            print(f"      Text: {payload.get('text', 'N/A')[:50]}...")

        # Test filtering
        print("\n6. Testing filtered search...")
        filter_dict = {"source": "test"}
        filtered_results = store.search(query_vector, limit=2, filter_dict=filter_dict)
        print(f"   Found {len(filtered_results)} vectors with filter source='test'")

        # Get updated collection info
        print("\n7. Getting updated collection info...")
        info = store.get_collection_info()
        print(f"   Total vectors: {info.get('vectors_count', 0)}")
        print(f"   Indexed vectors: {info.get('indexed_vectors_count', 0)}")

        print("\n" + "=" * 60)
        print("✓ All Qdrant tests passed successfully!")
        print("=" * 60)

        return True

    except Exception as e:
        print(f"\n✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_migration_capability():
    """Test ability to migrate from FAISS to Qdrant"""
    print("\n" + "=" * 60)
    print("Testing FAISS to Qdrant Migration Capability")
    print("=" * 60)

    # Check if FAISS index exists
    index_dir = Path("_index")
    if not index_dir.exists():
        print("No FAISS index found at _index/")
        print("Run 'python processing/processor.py embed --root .' to create FAISS index first")
        return False

    embeddings_path = index_dir / "embeddings.npy"
    mapping_path = index_dir / "mapping.json"

    if not embeddings_path.exists() or not mapping_path.exists():
        print("Required files not found:")
        print(f"  embeddings.npy: {embeddings_path.exists()}")
        print(f"  mapping.json: {mapping_path.exists()}")
        return False

    print(f"✓ Found FAISS index files at {index_dir}")
    print(f"  embeddings.npy: {embeddings_path.stat().st_size / 1024 / 1024:.2f} MB")
    print(f"  mapping.json: {mapping_path.stat().st_size / 1024:.2f} KB")

    print("\nTo migrate, you can use:")
    print("  store = QdrantVectorStore('config/qdrant_config.yaml')")
    print("  count = store.migrate_from_faiss('_index')")

    return True


if __name__ == "__main__":
    # Run tests
    success = test_qdrant_operations()

    if success:
        # Check migration capability
        test_migration_capability()

    sys.exit(0 if success else 1)
