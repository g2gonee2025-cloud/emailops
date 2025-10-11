# Qdrant Vector Database Setup for EmailOps

This document provides comprehensive information about the Qdrant vector database integration with the EmailOps project.

## Overview

Qdrant is a high-performance vector similarity search engine that provides an alternative to FAISS for storing and searching email embeddings. It offers:

- REST and gRPC APIs for easy integration
- Persistent storage with automatic backups
- Advanced filtering capabilities
- Horizontal scaling support
- Production-ready deployment options

## Current Status

✅ **Qdrant is already installed and running** via Docker on port 6333/6334

## Installation Details

### Docker Installation (Already Completed)

Qdrant is running as a Docker container with the following specifications:

- **Container Name**: qdrant
- **Image**: qdrant/qdrant:v1.7.4
- **HTTP Port**: 6333 (REST API)
- **gRPC Port**: 6334
- **Status**: Running and healthy

### Verify Installation

To check if Qdrant is running:

```bash
# Check Docker container status
docker ps | grep qdrant

# Test API endpoint
curl http://localhost:6333/

# Expected response:
# {"title":"qdrant - vector search engine","version":"1.7.4"}
```

## Configuration

### Configuration File Location

The Qdrant configuration for EmailOps is stored at:
```
config/qdrant_config.yaml
```

### Key Configuration Settings

- **Host**: localhost
- **Port**: 6333 (HTTP), 6334 (gRPC)
- **Collection Name**: emailops_embeddings
- **Vector Size**: 768 (default for most embedding models)
- **Distance Metric**: Cosine (best for normalized embeddings)
- **Batch Size**: 100 (for bulk operations)

### Environment Variables (Optional)

You can override configuration using environment variables:

```bash
export QDRANT_HOST=localhost
export QDRANT_PORT=6333
export QDRANT_COLLECTION=emailops_embeddings
export EMBEDDING_DIM=768
export QDRANT_DISTANCE=Cosine
```

## Usage

### Python Client

The EmailOps project includes a Qdrant client wrapper at `emailops/qdrant_client.py`.

#### Basic Usage

```python
from emailops.qdrant_client import QdrantVectorStore

# Initialize client
store = QdrantVectorStore("config/qdrant_config.yaml")

# Create collection (if needed)
store.create_collection()

# Insert embeddings
import numpy as np
embeddings = np.random.randn(10, 768).astype(np.float32)
metadata = [{"doc_id": f"doc_{i}", "text": f"Document {i}"} for i in range(10)]
count = store.upsert_embeddings(embeddings, metadata)

# Search for similar vectors
query_vector = np.random.randn(768).astype(np.float32)
results = store.search(query_vector, limit=5)
for doc_id, score, payload in results:
    print(f"ID: {doc_id}, Score: {score:.4f}, Text: {payload['text']}")
```

### Testing the Integration

Run the test script to verify everything is working:

```bash
python test_qdrant_integration.py
```

This will:
1. Test connection to Qdrant
2. Create the collection if it doesn't exist
3. Insert test vectors
4. Perform similarity search
5. Test filtered search

## Managing Qdrant

### Starting Qdrant

If Qdrant is not running, start it with:

```bash
# Using Docker (recommended)
docker run -p 6333:6333 -p 6334:6334 \
    -v ./qdrant_storage:/qdrant/storage \
    --name qdrant \
    --restart always \
    -d qdrant/qdrant:v1.7.4
```

### Stopping Qdrant

```bash
docker stop qdrant
```

### Restarting Qdrant

```bash
docker restart qdrant
```

### Removing Qdrant Container

```bash
# Stop and remove container
docker stop qdrant
docker rm qdrant

# Remove data (WARNING: This deletes all stored vectors)
rm -rf ./qdrant_storage
```

## Migration from FAISS

If you have existing FAISS indexes, you can migrate them to Qdrant:

```python
from emailops.qdrant_client import QdrantVectorStore

# Initialize Qdrant
store = QdrantVectorStore("config/qdrant_config.yaml")

# Migrate from FAISS index directory
count = store.migrate_from_faiss("_index")
print(f"Migrated {count} vectors to Qdrant")
```

## API Access

### REST API

Qdrant provides a REST API at `http://localhost:6333`:

```bash
# Get collections
curl http://localhost:6333/collections

# Get collection info
curl http://localhost:6333/collections/emailops_embeddings

# Get cluster info
curl http://localhost:6333/cluster
```

### Web UI

Qdrant includes a web dashboard accessible at:
```
http://localhost:6333/dashboard
```

## Troubleshooting

### Connection Issues

1. **Check if Docker is running**:
   ```bash
   docker --version
   systemctl status docker  # Linux
   ```

2. **Check if Qdrant container is running**:
   ```bash
   docker ps | grep qdrant
   ```

3. **Check logs**:
   ```bash
   docker logs qdrant
   ```

### Version Compatibility Warning

If you see a warning about version compatibility:
```
Qdrant client version X.X.X is incompatible with server version Y.Y.Y
```

This is usually harmless for minor version differences. To suppress:

```python
from qdrant_client import QdrantClient
client = QdrantClient(host="localhost", port=6333, check_compatibility=False)
```

### Performance Tuning

For large-scale deployments:

1. **Increase memory allocation**:
   ```bash
   docker run -p 6333:6333 -m 4g qdrant/qdrant
   ```

2. **Enable on-disk payload storage** (configured in qdrant_config.yaml)

3. **Adjust optimization settings** in the configuration file

## Integration with EmailOps Workflow

### Indexing Workflow

1. Process emails and generate chunks
2. Create embeddings using your preferred model
3. Store embeddings in Qdrant:
   ```python
   store.upsert_embeddings(embeddings, metadata)
   ```

### Search Workflow

1. Generate embedding for query
2. Search Qdrant for similar vectors:
   ```python
   results = store.search(query_embedding, limit=10)
   ```
3. Retrieve and rank results

### Filtering

Qdrant supports advanced filtering:

```python
# Search only in specific email accounts
results = store.search(
    query_vector,
    limit=10,
    filter_dict={"account": "user@example.com"}
)

# Search by date range
results = store.search(
    query_vector,
    limit=10,
    filter_dict={"date": {"$gte": "2025-01-01"}}
)
```

## Best Practices

1. **Normalize embeddings** before storing (cosine similarity works best with normalized vectors)
2. **Use batch operations** for inserting large numbers of vectors
3. **Create indexes** after bulk loading for better search performance
4. **Monitor collection size** and optimize when needed
5. **Regular backups** of the Qdrant storage directory

## Additional Resources

- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [Qdrant Python Client](https://github.com/qdrant/qdrant-client)
- [Qdrant REST API Reference](https://qdrant.github.io/qdrant/redoc/index.html)
- [Performance Tuning Guide](https://qdrant.tech/documentation/performance/)

## Summary

Qdrant is fully integrated with EmailOps and provides a robust, scalable solution for vector similarity search. The Docker-based setup ensures easy deployment and management, while the Python client wrapper simplifies integration with existing EmailOps workflows.