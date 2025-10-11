"""
Qdrant Vector Database Client for EmailOps
Provides integration between EmailOps and Qdrant vector database
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, TYPE_CHECKING

import numpy as np
import yaml

# Qdrant imports
if TYPE_CHECKING:
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance,
        FieldCondition,
        Filter,
        MatchValue,
        PointStruct,
        UpdateStatus,
        VectorParams,
    )

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance,
        FieldCondition,
        Filter,
        MatchValue,
        PointStruct,
        UpdateStatus,
        VectorParams,
    )
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    class MockQdrantClient:
        def __init__(self, **_): pass
        def get_collections(self, **_): return self
        @property
        def collections(self): return []
        def delete_collection(self, **_): pass
        def create_collection(self, **_): pass
        def upsert(self, **_): return self
        @property
        def status(self): return 'ok'
        def search(self, **_): return []
        def get_collection(self, **_): return self
        @property
        def config(self): return self
        @property
        def params(self): return self
        @property
        def vectors(self): return self
        @property
        def size(self): return 0
        @property
        def distance(self): return ''
        @property
        def vectors_count(self): return 0
        @property
        def points_count(self): return 0
        @property
        def indexed_vectors_count(self): return 0

    class MockDistance:
        COSINE = 'Cosine'
        EUCLID = 'Euclid'
        DOT = 'Dot'

    class MockUpdateStatus:
        COMPLETED = 'completed'

    class MockFieldCondition:
        def __init__(self, **_): pass

    class MockFilter:
        def __init__(self, **_): pass

    class MockMatchValue:
        def __init__(self, **_): pass

    class MockPointStruct:
        def __init__(self, **_): pass

    class MockVectorParams:
        def __init__(self, **_): pass

    QdrantClient = MockQdrantClient
    Distance = MockDistance
    FieldCondition = MockFieldCondition
    Filter = MockFilter
    MatchValue = MockMatchValue
    PointStruct = MockPointStruct
    UpdateStatus = MockUpdateStatus
    VectorParams = MockVectorParams
    print("Warning: qdrant-client not installed. Install with: pip install qdrant-client")

logger = logging.getLogger(__name__)


class QdrantVectorStore:
    """Qdrant vector store implementation for EmailOps"""

    def __init__(self, config_path: str | None = None):
        """
        Initialize Qdrant client with configuration

        Args:
            config_path: Path to qdrant_config.yaml file
        """
        if not QDRANT_AVAILABLE:
            raise ImportError("qdrant-client is not installed. Install with: pip install qdrant-client")

        # Load configuration
        self.config = self._load_config(config_path)

        # Initialize client
        self.client = QdrantClient(
            host=self.config['qdrant']['host'],
            port=self.config['qdrant']['port'],
            api_key=self.config['qdrant']['api_key'],
            https=self.config['qdrant']['https']
        )

        self.collection_name = self.config['qdrant']['collection_name']
        self.vector_size = self.config['qdrant']['vector_config']['size']
        self.distance_metric = self._get_distance_metric(
            self.config['qdrant']['vector_config']['distance']
        )

        logger.info(f"Initialized Qdrant client - host: {self.config['qdrant']['host']}:{self.config['qdrant']['port']}")

    def _load_config(self, config_path: str | None = None) -> dict[str, Any]:
        """Load configuration from YAML file or use defaults"""
        if config_path is None:
            # Look for config in standard locations
            possible_paths = [
                Path("config/qdrant_config.yaml"),
                Path("qdrant_config.yaml"),
                Path(".qdrant_config.yaml"),
            ]
            for path in possible_paths:
                if path.exists():
                    config_path = str(path)
                    break

        if config_path and Path(config_path).exists():
            with Path(config_path).open() as f:
                return yaml.safe_load(f)
        else:
            # Return default configuration
            return {
                'qdrant': {
                    'host': os.getenv('QDRANT_HOST', 'localhost'),
                    'port': int(os.getenv('QDRANT_PORT', 6333)),
                    'api_key': os.getenv('QDRANT_API_KEY'),
                    'https': os.getenv('QDRANT_HTTPS', 'false').lower() == 'true',
                    'collection_name': os.getenv('QDRANT_COLLECTION', 'emailops_embeddings'),
                    'vector_config': {
                        'size': int(os.getenv('EMBEDDING_DIM', 768)),
                        'distance': os.getenv('QDRANT_DISTANCE', 'Cosine')
                    }
                }
            }

    def _get_distance_metric(self, distance_name: str) -> Distance:
        """Convert string distance name to Qdrant Distance enum"""
        distance_map = {
            'Cosine': Distance.COSINE,
            'Euclid': Distance.EUCLID,
            'Dot': Distance.DOT,
        }
        return distance_map.get(distance_name, Distance.COSINE)

    def create_collection(self, recreate: bool = False) -> bool:
        """
        Create a new collection in Qdrant

        Args:
            recreate: If True, delete existing collection and create new one

        Returns:
            True if collection was created, False if it already exists
        """
        try:
            # Check if collection exists
            collections = self.client.get_collections()
            exists = any(col.name == self.collection_name for col in collections.collections)

            if exists:
                if recreate:
                    logger.info(f"Deleting existing collection: {self.collection_name}")
                    self.client.delete_collection(self.collection_name)
                else:
                    logger.info(f"Collection {self.collection_name} already exists")
                    return False

            # Create new collection
            logger.info(f"Creating collection: {self.collection_name}")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=self.distance_metric
                )
            )

            logger.info(f"Collection {self.collection_name} created successfully")
            return True

        except Exception as e:
            logger.error(f"Error creating collection: {e}")
            raise

    def upsert_embeddings(
        self,
        embeddings: np.ndarray,
        metadata: list[dict[str, Any]],
        batch_size: int | None = None
    ) -> int:
        """
        Insert or update embeddings in Qdrant

        Args:
            embeddings: Numpy array of embeddings (N x D)
            metadata: List of metadata dictionaries for each embedding
            batch_size: Batch size for insertion (default from config)

        Returns:
            Number of points inserted
        """
        if len(embeddings) != len(metadata):
            raise ValueError(f"Embeddings count ({len(embeddings)}) != metadata count ({len(metadata)})")

        effective_batch_size = batch_size or self.config['qdrant'].get('batch_size', 100)
        total_inserted = 0

        # Process in batches
        for i in range(0, len(embeddings), effective_batch_size):
            batch_end = min(i + effective_batch_size, len(embeddings))
            batch_embeddings = embeddings[i:batch_end]
            batch_metadata = metadata[i:batch_end]

            points = []
            for j, (embedding, meta) in enumerate(zip(batch_embeddings, batch_metadata, strict=False)):
                # Use the document ID if available, otherwise generate one
                point_id = meta.get('id', i + j)

                # Convert numpy array to list
                vector = embedding.tolist() if isinstance(embedding, np.ndarray) else embedding

                points.append(
                    PointStruct(
                        id=point_id,
                        vector=vector,
                        payload=meta
                    )
                )

            # Upsert batch
            operation_info = self.client.upsert(
                collection_name=self.collection_name,
                wait=True,
                points=points
            )

            if operation_info.status == UpdateStatus.COMPLETED:
                total_inserted += len(points)
                logger.info(f"Inserted batch {i//effective_batch_size + 1}: {len(points)} points")
            else:
                logger.error(f"Failed to insert batch {i//effective_batch_size + 1}")

        logger.info(f"Total points inserted: {total_inserted}")
        return total_inserted

    def search(
        self,
        query_vector: np.ndarray,
        limit: int = 10,
        filter_dict: dict[str, Any] | None = None,
        with_payload: bool = True,
        with_vectors: bool = False
    ) -> list[tuple[int, float, dict[str, Any]]]:
        """
        Search for similar vectors in Qdrant

        Args:
            query_vector: Query embedding vector
            limit: Number of results to return
            filter_dict: Optional filter conditions
            with_payload: Include payload in results
            with_vectors: Include vectors in results

        Returns:
            List of tuples (id, score, payload)
        """
        # Convert numpy array to list
        if isinstance(query_vector, np.ndarray):
            query_vector = query_vector.tolist()

        # Build filter if provided
        query_filter = None
        if filter_dict:
            conditions = []
            for key, value in filter_dict.items():
                conditions.append(
                    FieldCondition(
                        key=key,
                        match=MatchValue(value=value)
                    )
                )
            if conditions:
                query_filter = Filter(must=conditions)

        # Perform search
        search_result = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=limit,
            query_filter=query_filter,
            with_payload=with_payload,
            with_vectors=with_vectors
        )

        # Format results
        results = []
        for hit in search_result:
            results.append((
                hit.id,
                hit.score,
                hit.payload if with_payload else {}
            ))

        return results

    def get_collection_info(self) -> dict[str, Any]:
        """Get information about the collection"""
        try:
            collection = self.client.get_collection(self.collection_name)
            vector_params = collection.config.params.vectors
            return {
                'name': self.collection_name,
                'vectors_count': collection.vectors_count,
                'points_count': collection.points_count,
                'indexed_vectors_count': collection.indexed_vectors_count,
                'status': collection.status,
                'config': {
                    'vector_size': vector_params.size if vector_params else None,
                    'distance': str(vector_params.distance) if vector_params else None,
                }
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return {}

    def delete_collection(self) -> bool:
        """Delete the collection"""
        try:
            self.client.delete_collection(self.collection_name)
            logger.info(f"Collection {self.collection_name} deleted")
            return True
        except Exception as e:
            logger.error(f"Error deleting collection: {e}")
            return False

    def migrate_from_faiss(self, index_dir: str) -> int:
        """
        Migrate embeddings from FAISS index to Qdrant

        Args:
            index_dir: Directory containing FAISS index files

        Returns:
            Number of vectors migrated
        """
        index_path = Path(index_dir)
        embeddings_path = index_path / "embeddings.npy"
        mapping_path = index_path / "mapping.json"

        if not embeddings_path.exists() or not mapping_path.exists():
            raise FileNotFoundError(f"Required files not found in {index_dir}")

        # Load embeddings
        logger.info(f"Loading embeddings from {embeddings_path}")
        embeddings = np.load(embeddings_path)

        # Load mapping
        logger.info(f"Loading mapping from {mapping_path}")
        with mapping_path.open() as f:
            mapping = json.load(f)

        if len(embeddings) != len(mapping):
            logger.warning(f"Embeddings count ({len(embeddings)}) != mapping count ({len(mapping)})")

        # Create collection
        self.create_collection(recreate=True)

        # Prepare metadata with IDs
        for i, item in enumerate(mapping):
            if 'id' not in item:
                item['id'] = i

        # Insert embeddings
        logger.info(f"Migrating {len(embeddings)} vectors to Qdrant")
        count = self.upsert_embeddings(embeddings, mapping)

        logger.info(f"Migration complete: {count} vectors migrated")
        return count


def test_qdrant_connection(config_path: str | None = None) -> bool:
    """
    Test Qdrant connection and basic operations

    Args:
        config_path: Path to configuration file

    Returns:
        True if connection successful
    """
    try:
        # Initialize client
        store = QdrantVectorStore(config_path)

        # Get collections
        info = store.get_collection_info()

        if info:
            print("✓ Connected to Qdrant")
            print(f"  Collection: {info.get('name', 'N/A')}")
            print(f"  Vectors: {info.get('vectors_count', 0)}")
            print(f"  Status: {info.get('status', 'unknown')}")
        else:
            print("✓ Connected to Qdrant (collection not found)")

        return True

    except Exception as e:
        print(f"✗ Failed to connect to Qdrant: {e}")
        return False


if __name__ == "__main__":
    # Test the connection
    test_qdrant_connection()
