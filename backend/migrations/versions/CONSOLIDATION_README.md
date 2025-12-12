# Migration Consolidation Notes

## Embedding Dimension Migrations (Consolidated Dec 2025)

The following migrations were consolidated into `005_consolidated_embedding_dim.py`:

| Original Migration | Dimension Change | Reason |
|-------------------|------------------|--------|
| `005_update_embedding_dim.py` | 1536 → 3072 | Gemini embeddings |
| `006_add_hnsw_index.py` | (index only) | HNSW index creation |
| `007_resize_embedding_dim_3840.py` | 3072 → 3840 | Dimension increase |
| `008_resize_embedding_dim_1024.py` | 3840 → 1024 | BAAI/bge-m3 model |
| `009_resize_embedding_dim_3840.py` | 1024 → 3840 | KaLM model |

### Net Effect
The consolidated migration goes directly from the initial dimension (1536) to the final dimension (3840), skipping all intermediate steps.

### For Existing Deployments
If your database has already run through migrations 005-009:
1. The current state is already 3840 dimensions
2. You can manually update `alembic_version` to `005_consolidated_embedding_dim`
3. Or simply leave it at `009_resize_embedding_dim_3840` (the end state is identical)

### For Fresh Deployments
Run migrations normally - the consolidated migration will set up 3840 dimensions directly.

### Superseded Files
The `.superseded` files are kept for reference but will not be executed by Alembic.
