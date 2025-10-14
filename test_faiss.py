#!/usr/bin/env python3
"""Test FAISS installation and functionality"""

import sys

import numpy as np

print("Testing FAISS installation and functionality...")
print("=" * 60)

# Test 1: Import FAISS
try:
    import faiss
    print("✅ FAISS import successful")
    print(f"   FAISS version info available attributes: {[attr for attr in dir(faiss) if 'version' in attr.lower()]}")
except ImportError as e:
    print(f"❌ FAISS import failed: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Unexpected error importing FAISS: {e}")
    sys.exit(1)

# Test 2: Create a simple index
try:
    print("\nTesting FAISS index creation...")

    # Create some test embeddings
    dim = 768  # Common embedding dimension (like for Gemini/BERT)
    n_vectors = 100

    # Create random embeddings
    embeddings = np.random.random((n_vectors, dim)).astype('float32')
    print(f"✅ Created test embeddings: shape={embeddings.shape}, dtype={embeddings.dtype}")

    # Create FAISS index
    index = faiss.IndexFlatIP(dim)  # Inner product similarity
    print(f"✅ Created FAISS IndexFlatIP with dimension {dim}")

    # Normalize embeddings
    embeddings_normalized = embeddings.copy()
    faiss.normalize_L2(embeddings_normalized)
    print("✅ Normalized embeddings for L2")

    # Ensure C-contiguous array
    embeddings_contiguous = np.ascontiguousarray(embeddings_normalized, dtype=np.float32)
    print("✅ Ensured C-contiguous array")

    # Add to index
    index.add(embeddings_contiguous)
    print(f"✅ Added {index.ntotal} vectors to index")

    # Test search
    query = embeddings_normalized[:5]  # Use first 5 as queries
    k = 10
    distances, indices = index.search(query, k)
    print(f"✅ Search successful: found {k} nearest neighbors for {len(query)} queries")

    # Test writing and reading index
    import os
    import tempfile

    with tempfile.NamedTemporaryFile(suffix='.faiss', delete=False) as tmp:
        tmp_path = tmp.name

    try:
        faiss.write_index(index, tmp_path)
        print(f"✅ Successfully wrote index to {tmp_path}")

        # Read it back
        index_loaded = faiss.read_index(tmp_path)
        print(f"✅ Successfully read index back: {index_loaded.ntotal} vectors")

    finally:
        # Clean up
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
            print("✅ Cleaned up temporary file")

    print("\n" + "=" * 60)
    print("✅ ALL FAISS TESTS PASSED SUCCESSFULLY!")

except Exception as e:
    print(f"\n❌ FAISS test failed with error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Check processor.py FAISS usage
print("\n" + "=" * 60)
print("Checking processor.py FAISS integration...")

try:
    from pathlib import Path

    # Check if there's an existing index
    index_paths = [
        Path("C:/Users/ASUS/Desktop/OUTLOOK/_index/index.faiss"),
        Path("C:/Users/ASUS/Desktop/OUTLOOK_index/index.faiss"),
    ]

    for index_path in index_paths:
        if index_path.exists():
            print(f"✅ Found existing FAISS index at: {index_path}")
            try:
                test_index = faiss.read_index(str(index_path))
                print(f"   Index contains {test_index.ntotal} vectors")
                print(f"   Index dimension: {test_index.d}")
            except Exception as e:
                print(f"   ⚠️ Could not read index: {e}")
        else:
            print(f"ℹ️ No index found at: {index_path}")

    # Check embeddings file
    emb_paths = [
        Path("C:/Users/ASUS/Desktop/OUTLOOK/_index/embeddings.npy"),
        Path("C:/Users/ASUS/Desktop/OUTLOOK_index/embeddings.npy"),
    ]

    for emb_path in emb_paths:
        if emb_path.exists():
            print(f"✅ Found embeddings file at: {emb_path}")
            try:
                embs = np.load(str(emb_path))
                print(f"   Embeddings shape: {embs.shape}")
                print(f"   Embeddings dtype: {embs.dtype}")
            except Exception as e:
                print(f"   ⚠️ Could not read embeddings: {e}")

except Exception as e:
    print(f"⚠️ Error checking existing files: {e}")

print("\n" + "=" * 60)
print("FAISS diagnostic complete!")
