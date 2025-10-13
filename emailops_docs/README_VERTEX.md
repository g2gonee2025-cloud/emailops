# EmailOps Vertex AI Implementation Guide

> **Version:** 1.0  
> **Provider:** Vertex AI (Google Cloud)  
> **Model:** gemini-embedding-001 (default)  
> **Status:** Production-Ready ✅

---

## Quick Start

### Prerequisites

1. **Google Cloud Project** with Vertex AI enabled
2. **Service Account JSON** with appropriate permissions
3. **Python 3.10+** with required dependencies

### Installation

```bash
# Clone repository
git clone <repository_url>
cd emailops_vertex_ai

# Install dependencies
pip install -r requirements.txt

# Set up credentials
mkdir -p secrets/
cp /path/to/service-account.json secrets/

# Set environment variables
export GCP_PROJECT="your-project-id"
export GOOGLE_APPLICATION_CREDENTIALS="secrets/your-service-account.json"
```

### Basic Usage

```bash
# Build index (first time)
python -m emailops.email_indexer \
  --root /path/to/email/export \
  --provider vertex \
  --force-reindex

# Incremental update (subsequent runs)
python -m emailops.email_indexer \
  --root /path/to/email/export \
  --provider vertex

# Search the index
python -m emailops.search_and_draft \
  --root /path/to/email/export \
  --provider vertex \
  --query "insurance policy renewal"
```

---

## Architecture Overview

### Vertex-Only Design

EmailOps is built exclusively for **Vertex AI** embeddings, leveraging Google's Gemini models for semantic search and email processing. This focused approach enables:

- **Simplified codebase** — No multi-provider complexity
- **Optimized performance** — Tuned for Vertex AI specifics
- **Reliable credentials** — Centralized GCP authentication
- **Consistent dimensions** — 3072 for Gemini, 768 for legacy

### Key Components

```
emailops/
├── email_indexer.py      # Vector index builder (Vertex-only)
├── index_metadata.py     # Metadata & consistency management
├── config.py             # Centralized configuration
├── utils.py              # Text extraction & email cleaning
├── validators.py         # Security & input validation
├── text_chunker.py       # Chunking logic
├── llm_client.py         # Embedding client (shim)
└── llm_runtime.py        # Vertex AI runtime
```

### Data Flow

```
Email Export
    ↓
[Conversation Discovery] → find_conversation_dirs()
    ↓
[Text Extraction] → extract_text() + clean_email_text()
    ↓
[Chunking] → prepare_index_units()
    ↓
[Embedding] → embed_texts() via Vertex AI
    ↓
[Persistence] → embeddings.npy + mapping.json + index.faiss
    ↓
[Consistency Check] → validate counts & dimensions
    ↓
Vector Index (_index/)
```

---

## Configuration

### Required Environment Variables

```bash
# GCP Project (required)
export GCP_PROJECT="your-project-id"

# Service Account (auto-discovered from secrets/ or set explicitly)
export GOOGLE_APPLICATION_CREDENTIALS="secrets/service-account.json"

# Optional: Override defaults
export VERTEX_EMBED_MODEL="gemini-embedding-001"  # Default
export EMBED_BATCH="64"                           # Batch size (max 250)
export CHUNK_SIZE="1600"                          # Characters per chunk
export CHUNK_OVERLAP="200"                        # Overlap between chunks
```

### Configuration File

The [`config.py`](../emailops/config.py) module provides centralized configuration:

```python
from emailops.config import EmailOpsConfig

# Load configuration
config = EmailOpsConfig.load()

# Access settings
print(config.VERTEX_EMBED_MODEL)  # "gemini-embedding-001"
print(config.DEFAULT_CHUNK_SIZE)  # 1600
print(config.GCP_PROJECT)         # From environment

# Update environment for child processes
config.update_environment()
```

---

## Index Building

### Three Update Modes

EmailOps supports three strategies for building/updating the vector index:

#### 1. Full Rebuild (`--force-reindex`)

Use when:
- Building index for the first time
- Changing embedding model or dimensions
- Index corruption detected
- Major schema changes

```bash
python -m emailops.email_indexer \
  --root /path/to/export \
  --provider vertex \
  --force-reindex
```

#### 2. Timestamp-Based Incremental

Use when:
- Regular updates after initial build
- `file_times.json` not available
- Fast updates needed

```bash
python -m emailops.email_indexer \
  --root /path/to/export \
  --provider vertex
```

**How it works:**
- Uses `last_run.txt` as cutoff timestamp
- Reuses embeddings for files with `mtime < last_run`
- Re-embeds changed/new files only

#### 3. File-Times Incremental (Precise)

Use when:
- Maximum efficiency needed
- Deletions must be tracked
- `file_times.json` exists from prior run

```bash
python -m emailops.email_indexer \
  --root /path/to/export \
  --provider vertex
```

**How it works:**
- Uses per-document `mtime` from `file_times.json`
- Tracks adds, edits, AND deletions
- Most efficient for steady-state updates

### Advanced Options

```bash
# Custom model
python -m emailops.email_indexer \
  --root /path/to/export \
  --provider vertex \
  --model text-embedding-005

# Larger batch size (up to 250)
python -m emailops.email_indexer \
  --root /path/to/export \
  --provider vertex \
  --batch 128

# Separate index directory
python -m emailops.email_indexer \
  --root /path/to/export \
  --provider vertex \
  --index-root /path/to/custom/index

# Limit chunks per conversation (testing)
python -m emailops.email_indexer \
  --root /path/to/export \
  --provider vertex \
  --limit 10
```

---

## Index Artifacts

### Directory Structure

After building, the index directory contains:

```
_index/
├── embeddings.npy        # NumPy array: (N, D) float32
├── mapping.json          # N document metadata entries
├── index.faiss           # Optional FAISS index (IndexFlatIP)
├── meta.json             # Index metadata (provider, model, dims, counts)
├── file_times.json       # Per-document mtime for incremental updates
└── last_run.txt          # ISO 8601 timestamp of last build
```

### Artifact Details

**`embeddings.npy`:**
- Shape: (N documents, D dimensions)
- Dtype: float32
- Normalized: Unit vectors (L2 norm = 1)
- Size: ~N × D × 4 bytes

**`mapping.json`:**
- Schema: List of document metadata objects
- Fields: id, path, conv_id, doc_type, subject, date, snippet, etc.
- Size: ~1 KB per document

**`index.faiss`:**
- Type: IndexFlatIP (Inner Product for cosine similarity)
- Built: Only if faiss-cpu installed
- Size: ~N × D × 4 bytes + overhead

**`meta.json`:**
- Provider: "vertex"
- Model: e.g., "gemini-embedding-001"
- Dimensions: Configured (3072 for Gemini) and actual
- Counts: num_documents, num_folders
- Index type: "faiss", "numpy", or "none"

---

## Credentials & Authentication

### Service Account Setup

1. **Create Service Account:**
   ```bash
   gcloud iam service-accounts create emailops-indexer \
     --display-name="EmailOps Indexer"
   ```

2. **Grant Permissions:**
   ```bash
   gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
     --member="serviceAccount:emailops-indexer@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
     --role="roles/aiplatform.user"
   ```

3. **Create & Download Key:**
   ```bash
   gcloud iam service-accounts keys create secrets/emailops-sa.json \
     --iam-account=emailops-indexer@YOUR_PROJECT_ID.iam.gserviceaccount.com
   ```

### Auto-Discovery

EmailOps automatically discovers credentials in this order:

1. `GOOGLE_APPLICATION_CREDENTIALS` environment variable
2. Files in `secrets/` directory (priority order):
   - `api-agent-470921-aa03081a1b4d.json`
   - `apt-arcana-470409-i7-ce42b76061bf.json`
   - `crafty-airfoil-474021-s2-34159960925b.json`
   - `embed2-474114-fca38b4d2068.json`
   - `my-project-31635v-8ec357ac35b2.json`
   - `semiotic-nexus-470620-f3-3240cfaf6036.json`

To use custom credentials, either:
- Set `GOOGLE_APPLICATION_CREDENTIALS` environment variable, OR
- Place JSON file in `secrets/` with a name from the priority list, OR
- Update `CREDENTIAL_FILES_PRIORITY` in [`config.py`](../emailops/config.py:47)

---

## Embedding Models

### Supported Vertex Models

| Model | Dimensions | Use Case | Performance |
|-------|-----------|----------|-------------|
| `gemini-embedding-001` | 3072 | **Recommended** — Best quality | High |
| `text-embedding-005` | 768 | Legacy, smaller index | Medium |
| `text-embedding-004` | 768 | Legacy | Medium |
| `textembedding-gecko@latest` | 768 | Legacy | Low |
| `text-multilingual-embedding-*` | 768 | Multilingual | Medium |

### Model Selection

**Default (recommended):**
```bash
# Uses gemini-embedding-001 (3072 dims)
python -m emailops.email_indexer --root /path/to/export --provider vertex
```

**Override via CLI:**
```bash
python -m emailops.email_indexer \
  --root /path/to/export \
  --provider vertex \
  --model text-embedding-005
```

**Override via environment:**
```bash
export VERTEX_EMBED_MODEL="text-embedding-005"
python -m emailops.email_indexer --root /path/to/export --provider vertex
```

**⚠️ Important:** Changing models requires `--force-reindex` to rebuild with new dimensions.

---

## File Size Limits

### Default Limits

```bash
MAX_INDEXABLE_FILE_MB=50          # Skip files larger than 50 MB
MAX_INDEXABLE_CHARS=5000000       # Truncate text longer than 5M chars
MAX_ATTACHMENT_TEXT_CHARS=500000  # Max chars per attachment
EXCEL_MAX_CELLS=200000            # Max cells in Excel files
```

### Override Limits

```bash
# Index larger files
export MAX_INDEXABLE_FILE_MB=100
export MAX_INDEXABLE_CHARS=10000000

python -m emailops.email_indexer \
  --root /path/to/export \
  --provider vertex
```

### Behavior

When limits are exceeded:
- **Files > MAX_INDEXABLE_FILE_MB:** Skipped with warning log
- **Text > MAX_INDEXABLE_CHARS:** Truncated with warning log
- **Excel > EXCEL_MAX_CELLS:** Rows truncated to stay within limit

---

## Performance Optimization

### Batch Size Tuning

```bash
# Default: 64 texts per embedding call
python -m emailops.email_indexer --root /path/to/export --batch 64

# Higher throughput (if quota allows)
python -m emailops.email_indexer --root /path/to/export --batch 128

# Maximum (250 texts per call)
python -m emailops.email_indexer --root /path/to/export --batch 250
```

**Considerations:**
- Larger batches = fewer API calls = faster builds
- Limited by Vertex AI quotas (check Cloud Console)
- Max 250 enforced to prevent errors

### Incremental Updates

For large corpora (>5,000 conversations), prefer incremental mode:

```bash
# Initial build (full)
python -m emailops.email_indexer \
  --root /path/to/export \
  --force-reindex

# Daily updates (incremental)
python -m emailops.email_indexer \
  --root /path/to/export
```

**Performance:**
- Full build: ~20-30 min for 10k conversations
- Incremental: ~2-5 min for 1k changes
- File-times mode: ~1-3 min for 1k changes

---

## Troubleshooting

### Common Issues

#### "No index metadata found"

**Cause:** Index hasn't been built yet

**Solution:**
```bash
python -m emailops.email_indexer \
  --root /path/to/export \
  --provider vertex \
  --force-reindex
```

---

#### "Dimension mismatch: index has X dims, but Vertex config is Y"

**Cause:** Index was built with a different model/dimension setting

**Solution:**
```bash
# Option 1: Rebuild with current model
python -m emailops.email_indexer \
  --root /path/to/export \
  --provider vertex \
  --force-reindex

# Option 2: Set dimension explicitly
export VERTEX_OUTPUT_DIM=3072  # Match existing index
python -m emailops.email_indexer \
  --root /path/to/export \
  --provider vertex
```

---

#### "Failed to initialize GCP credentials"

**Cause:** Missing or invalid service account JSON

**Solution:**
```bash
# Check if file exists
ls -la secrets/

# Verify JSON structure
cat secrets/your-file.json | jq '.project_id, .client_email'

# Set explicitly
export GOOGLE_APPLICATION_CREDENTIALS="$(pwd)/secrets/your-file.json"
```

**Required JSON fields:**
- `type: "service_account"`
- `project_id`
- `private_key_id`
- `private_key`
- `client_email`

---

#### "Embedding provider returned empty vectors"

**Cause:** Quota exhausted, network issue, or permission problem

**Solution:**
```bash
# Check quotas in Cloud Console
# Navigate to: Vertex AI > Quotas

# Verify permissions
gcloud projects get-iam-policy YOUR_PROJECT_ID \
  --flatten="bindings[].members" \
  --filter="bindings.members:serviceAccount:YOUR_SA_EMAIL"

# Should include: roles/aiplatform.user

# Reduce batch size to stay under quota
python -m emailops.email_indexer \
  --root /path/to/export \
  --batch 32
```

---

#### "Embeddings/document count mismatch"

**Cause:** Index corruption or interrupted build

**Solution:**
```bash
# Rebuild from scratch
python -m emailops.email_indexer \
  --root /path/to/export \
  --provider vertex \
  --force-reindex
```

---

## Documentation

### Core Documentation

| Document | Description | Status |
|----------|-------------|--------|
| [`email_indexer.md`](email_indexer.md) | Index builder & update strategies | ✅ |
| [`index_metadata.md`](index_metadata.md) | Metadata management & validation | ✅ |
| [`utils.md`](utils.md) | Text extraction & email utilities | ✅ |
| [`validators.md`](validators.md) | Security & validation helpers | ✅ |

### Verification & Alignment

| Document | Description | Status |
|----------|-------------|--------|
| [`VERTEX_ALIGNMENT_SUMMARY.md`](VERTEX_ALIGNMENT_SUMMARY.md) | Comprehensive alignment report | ✅ |
| [`IMPLEMENTATION_VERIFICATION.md`](IMPLEMENTATION_VERIFICATION.md) | Detailed verification matrix | ✅ |
| **This document** | Quick start & operational guide | ✅ |

---

## API Reference

### Core Functions

#### `email_indexer.main()`

Build or update the vector index.

**CLI:**
```bash
python -m emailops.email_indexer --root <path> [options]
```

**Required:**
- `--root` — Export root containing conversation folders

**Optional:**
- `--provider` — Must be "vertex" (only allowed value)
- `--model` — Override VERTEX_EMBED_MODEL
- `--batch` — Embedding batch size (1-250, default 64)
- `--index-root` — Custom index directory
- `--force-reindex` — Full rebuild
- `--limit` — Cap chunks per conversation (testing)

---

#### `EmailOpsConfig.load()`

Load configuration from environment.

**Usage:**
```python
from emailops.config import EmailOpsConfig

config = EmailOpsConfig.load()
config.update_environment()  # Propagate to os.environ
```

**Returns:** Configured `EmailOpsConfig` instance with all settings

---

#### `embed_texts(texts, provider="vertex")`

Generate embeddings for a list of texts.

**Usage:**
```python
from emailops.llm_client import embed_texts

vectors = embed_texts(["Hello world", "Goodbye"], provider="vertex")
# Returns: np.ndarray shape (2, 3072) dtype float32, unit-normalized
```

**Parameters:**
- `texts` — List of strings to embed
- `provider` — Must be "vertex"

**Returns:** NumPy array (N, D) with unit-normalized vectors

---

#### `validate_index_compatibility(index_dir, provider="vertex")`

Verify index can be used with current configuration.

**Usage:**
```python
from emailops.index_metadata import validate_index_compatibility

is_valid = validate_index_compatibility("/path/to/_index", provider="vertex")
if not is_valid:
    print("Index incompatible; rebuild required")
```

**Checks:**
- Provider matches ("vertex")
- Artifacts exist (embeddings.npy or index.faiss)
- Dimensions match configuration
- Counts align across artifacts

---

## Best Practices

### 1. Credential Security

✅ **DO:**
- Store service account JSON in `secrets/` (gitignored)
- Use separate accounts for dev/staging/prod
- Rotate keys regularly
- Grant minimal required permissions

❌ **DON'T:**
- Commit credentials to version control
- Share service account keys
- Use personal GCP accounts for production
- Grant overly broad permissions

### 2. Index Maintenance

✅ **DO:**
- Run incremental updates regularly (daily/weekly)
- Monitor index size growth
- Back up `_index/` before major changes
- Use `--force-reindex` after model changes
- Verify consistency after updates

❌ **DON'T:**
- Edit index files manually
- Mix providers in same index
- Change dimensions without rebuild
- Delete artifacts piecemeal

### 3. Performance Tuning

✅ **DO:**
- Use largest batch size your quota allows
- Prefer file-times incremental for large corpora
- Monitor embedding call duration
- Scale vertically for faster builds (more RAM/CPU)

❌ **DON'T:**
- Set batch >250 (will be clamped)
- Run full rebuilds unnecessarily
- Skip consistency checks
- Ignore quota warnings

---

## Monitoring & Observability

### Key Metrics to Track

1. **Index Build Duration**
   - Full rebuild: Baseline for comparison
   - Incremental: Should be <10% of full build time

2. **Embedding Call Counts**
   - Reused: Should be >80% in incremental mode
   - New: Only changed/added documents

3. **Index Size**
   - embeddings.npy: N × D × 4 bytes
   - mapping.json: ~1 KB per document
   - Total: Track growth over time

4. **Error Rates**
   - Embedding failures: Should be <0.1%
   - Consistency check failures: Should be 0%
   - File read errors: Monitor and investigate

### Logging

```python
# Set log level
export LOG_LEVEL="DEBUG"  # For detailed diagnostics
export LOG_LEVEL="INFO"   # For normal operation (default)
export LOG_LEVEL="WARNING" # For quiet mode

python -m emailops.email_indexer --root /path/to/export
```

**Log Levels:**
- **DEBUG:** Dimension detection, score statistics
- **INFO:** Build progress, successful operations
- **WARNING:** Skipped files, fallbacks, non-critical issues
- **ERROR:** Critical failures, consistency check failures

---

## Production Deployment

### Checklist

- [ ] **GCP Project configured** with Vertex AI enabled
- [ ] **Service account created** with `roles/aiplatform.user`
- [ ] **Credentials placed** in `secrets/` directory
- [ ] **Environment variables set** (GCP_PROJECT, etc.)
- [ ] **Dependencies installed** (`pip install -r requirements.txt`)
- [ ] **Initial index built** with `--force-reindex`
- [ ] **Incremental updates tested** (add/edit/delete scenarios)
- [ ] **Consistency validation** passing
- [ ] **FAISS index built** (optional but recommended)
- [ ] **Backup strategy** for `_index/` directory
- [ ] **Monitoring configured** (logs, metrics)
- [ ] **Quota limits** checked and documented

### Example Production Script

```bash
#!/bin/bash
# production_index_build.sh

set -euo pipefail

# Configuration
export GCP_PROJECT="your-production-project"
export GOOGLE_APPLICATION_CREDENTIALS="$(pwd)/secrets/prod-service-account.json"
export LOG_LEVEL="INFO"
export EMBED_BATCH="128"

# Paths
EXPORT_ROOT="/data/email_exports"
INDEX_ROOT="/data/indexes"
BACKUP_DIR="/data/backups/indexes"

# Backup existing index
if [ -d "${INDEX_ROOT}/_index" ]; then
  BACKUP_NAME="index_backup_$(date +%Y%m%d_%H%M%S).tar.gz"
  tar -czf "${BACKUP_DIR}/${BACKUP_NAME}" -C "${INDEX_ROOT}" "_index"
  echo "Backed up existing index to ${BACKUP_NAME}"
fi

# Run incremental update
python -m emailops.email_indexer \
  --root "${EXPORT_ROOT}" \
  --provider vertex \
  --index-root "${INDEX_ROOT}" \
  --batch 128

# Verify consistency
python -c "
from emailops.index_metadata import check_index_consistency
try:
    check_index_consistency('${INDEX_ROOT}/_index', raise_on_mismatch=True)
    print('✅ Index consistency check passed')
except Exception as e:
    print(f'❌ Index consistency check failed: {e}')
    exit(1)
"

echo "✅ Index build completed successfully"
```

---

## Frequently Asked Questions

### Q: Why Vertex AI only?

**A:** Focusing on a single provider:
- Simplifies codebase and maintenance
- Enables provider-specific optimizations
- Ensures consistent dimensions across runs
- Leverages Google's latest Gemini models
- Reduces configuration complexity

### Q: Can I use other embedding providers?

**A:** No, the current implementation is constrained to Vertex AI. The CLI restricts `--provider` to `["vertex"]`. To support other providers, you would need to:
1. Remove CLI constraint
2. Add multi-provider dimension handling
3. Update metadata validation logic
4. Test compatibility thoroughly

### Q: What happens if I change the embedding model?

**A:** You must rebuild the index with `--force-reindex`. Different models have different dimensions (Gemini: 3072, text-embedding: 768), and mixing dimensions will cause errors.

### Q: How do I know which update mode is being used?

**A:** Check the logs:
- **Full:** "Starting full corpus scan"
- **Timestamp:** "Starting incremental (timestamp) update from YYYY-MM-DD"
- **File-times:** "Incremental corpus: X new/updated, Y unchanged, Z deleted"

### Q: Is FAISS required?

**A:** No, FAISS is optional but recommended. Without it:
- Index still works (uses `embeddings.npy`)
- Search is slower (brute-force cosine similarity)
- No approximate nearest neighbor speedup

Install FAISS: `pip install faiss-cpu`

### Q: How do I handle quota limits?

**A:** Reduce batch size and/or implement rate limiting:
```bash
# Smaller batches
python -m emailops.email_indexer --root /path --batch 32

# Add rate limiting via environment
export API_RATE_LIMIT="30"  # Calls per minute
python -m emailops.email_indexer --root /path
```

### Q: Can I run multiple indexers in parallel?

**A:** Not recommended for the same export directory. The indexer is designed for sequential updates. For parallel processing:
- Use separate export directories
- Write to separate index directories
- Merge indexes post-build (manual process)

---

## Migration Guide

### From Multi-Provider Setup

If you have an existing index built with a different provider:

1. **Backup existing index:**
   ```bash
   tar -czf index_backup.tar.gz _index/
   ```

2. **Rebuild with Vertex:**
   ```bash
   python -m emailops.email_indexer \
     --root /path/to/export \
     --provider vertex \
     --force-reindex
   ```

3. **Verify:**
   ```bash
   python -c "from emailops.index_metadata import get_index_info; print(get_index_info('_index'))"
   ```

### From Older EmailOps Version

If upgrading from a version without file-times tracking:

1. **First run after upgrade:**
   ```bash
   # Will use timestamp-based incremental
   python -m emailops.email_indexer --root /path/to/export
   ```

2. **Subsequent runs:**
   ```bash
   # Will use file-times incremental (preferred)
   python -m emailops.email_indexer --root /path/to/export
   ```

---

## Support & Resources

### Documentation Links

- **[Email Indexer](email_indexer.md)** — Detailed indexer documentation
- **[Index Metadata](index_metadata.md)** — Metadata & validation
- **[Utils](utils.md)** — Text extraction & cleaning
- **[Validators](validators.md)** — Security & validation
- **[Alignment Summary](VERTEX_ALIGNMENT_SUMMARY.md)** — Architecture overview
- **[Implementation Verification](IMPLEMENTATION_VERIFICATION.md)** — Verification report

### Code References

- **Index Builder:** [`emailops/email_indexer.py`](../emailops/email_indexer.py)
- **Configuration:** [`emailops/config.py`](../emailops/config.py)
- **Metadata Manager:** [`emailops/index_metadata.py`](../emailops/index_metadata.py)
- **Text Utilities:** [`emailops/utils.py`](../emailops/utils.py)
- **Security Validation:** [`emailops/validators.py`](../emailops/validators.py)

### External Resources

- **Vertex AI Embeddings:** https://cloud.google.com/vertex-ai/docs/generative-ai/embeddings/get-text-embeddings
- **Gemini Models:** https://cloud.google.com/vertex-ai/docs/generative-ai/learn/models
- **Service Accounts:** https://cloud.google.com/iam/docs/service-accounts
- **FAISS:** https://github.com/facebookresearch/faiss

---

## Contributing

### Code Style

- Follow PEP 8 guidelines
- Use type hints for all functions
- Write docstrings for public APIs
- Add inline comments for complex logic
- Keep functions focused and testable

### Testing

Before submitting changes:

1. **Run unit tests:**
   ```bash
   pytest tests/unit/
   ```

2. **Test full index build:**
   ```bash
   python -m emailops.email_indexer --root test_data --force-reindex
   ```

3. **Test incremental update:**
   ```bash
   python -m emailops.email_indexer --root test_data
   ```

4. **Verify consistency:**
   ```bash
   python -c "from emailops.index_metadata import check_index_consistency; \
              check_index_consistency('test_data/_index')"
   ```

---

## License

[Include your license information here]

---

## Contact

[Include contact/support information here]

---

**Last Updated:** 2025-01-12  
**Version:** 1.0 (Vertex-Only Implementation)