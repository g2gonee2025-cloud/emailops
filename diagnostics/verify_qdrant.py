#!/usr/bin/env python3
"""Verify Qdrant vector database connectivity and functionality."""

import builtins
import contextlib
import json
from datetime import datetime

import requests
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams


def check_qdrant_health():
    """Check if Qdrant is healthy via HTTP API."""
    try:
        response = requests.get("http://localhost:6333/")
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, f"HTTP {response.status_code}: {response.text}"
    except Exception as e:
        return False, str(e)

def test_qdrant_client():
    """Test Qdrant client operations."""
    results = {
        "connection": False,
        "list_collections": False,
        "create_collection": False,
        "insert_points": False,
        "search_points": False,
        "delete_collection": False,
        "errors": []
    }

    try:
        # Initialize client
        client = QdrantClient(host="localhost", port=6333)
        results["connection"] = True

        # List collections
        collections = client.get_collections()
        results["list_collections"] = True
        print(f"✓ Listed collections: {len(collections.collections)} found")

        # Create test collection
        test_collection = "test_verification_collection"

        # Delete if exists
        with contextlib.suppress(builtins.BaseException):
            client.delete_collection(test_collection)

        client.create_collection(
            collection_name=test_collection,
            vectors_config=VectorParams(size=4, distance=Distance.DOT),
        )
        results["create_collection"] = True
        print(f"✓ Created test collection: {test_collection}")

        # Insert test points
        points = [
            PointStruct(
                id=1,
                vector=[0.05, 0.61, 0.76, 0.74],
                payload={
                    "city": "Berlin",
                    "country": "Germany",
                    "population": 3700000,
                    "info": "Capital of Germany"
                }
            ),
            PointStruct(
                id=2,
                vector=[0.18, 0.01, 0.85, 0.80],
                payload={
                    "city": "London",
                    "country": "UK",
                    "population": 9000000,
                    "info": "Capital of UK"
                }
            ),
            PointStruct(
                id=3,
                vector=[0.24, 0.18, 0.22, 0.44],
                payload={
                    "city": "Paris",
                    "country": "France",
                    "population": 2200000,
                    "info": "Capital of France"
                }
            ),
        ]

        client.upsert(
            collection_name=test_collection,
            points=points
        )
        results["insert_points"] = True
        print(f"✓ Inserted {len(points)} test points")

        # Search for similar vectors
        search_result = client.search(
            collection_name=test_collection,
            query_vector=[0.2, 0.1, 0.9, 0.7],
            limit=2
        )
        results["search_points"] = True
        print(f"✓ Search successful, found {len(search_result)} results")

        # Clean up
        client.delete_collection(test_collection)
        results["delete_collection"] = True
        print("✓ Cleaned up test collection")

    except Exception as e:
        results["errors"].append(str(e))
        print(f"✗ Error: {e}")

    return results

def main():
    """Main verification function."""
    print("=" * 60)
    print("Qdrant Vector Database Verification")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    print()

    # Check health endpoint
    print("Checking Qdrant health...")
    print("-" * 60)
    health_ok, health_info = check_qdrant_health()

    if health_ok:
        print("✓ Qdrant is healthy")
        print(f"  Version: {health_info.get('version', 'Unknown')}")
        print(f"  Title: {health_info.get('title', 'Unknown')}")
    else:
        print(f"✗ Qdrant health check failed: {health_info}")

    print()

    # Test client operations
    print("Testing Qdrant client operations...")
    print("-" * 60)
    test_results = test_qdrant_client()

    # Summary
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Health Check: {'✓ PASSED' if health_ok else '✗ FAILED'}")
    print()
    print("Client Operations:")
    for operation, success in test_results.items():
        if operation != "errors":
            status = "✓ PASSED" if success else "✗ FAILED"
            print(f"  {operation:<20} {status}")

    # Overall status
    all_passed = health_ok and all(
        v for k, v in test_results.items() if k != "errors" and v is not False
    )

    print()
    print(f"Overall Status: {'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}")

    if test_results["errors"]:
        print("\nErrors encountered:")
        for error in test_results["errors"]:
            print(f"  - {error}")

    # Save results
    results = {
        "timestamp": datetime.now().isoformat(),
        "health_check": {
            "status": "OK" if health_ok else "FAILED",
            "info": health_info if health_ok else str(health_info)
        },
        "client_tests": test_results,
        "overall_status": "PASSED" if all_passed else "FAILED"
    }

    with open('qdrant_verification_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\nResults saved to: qdrant_verification_results.json")

if __name__ == "__main__":
    main()
