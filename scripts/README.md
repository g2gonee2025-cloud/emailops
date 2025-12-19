
# Scripts Directory

This directory contains utility scripts for the EmailOps system, organized by function.

## 📂 Structure

### `search/`
- **`search_cli.py`**: Unified search tool using the production backend.
  - Usage: `python scripts/search/search_cli.py "query" --limit 5`

### `verification/`
- **`inspect_db.py`**: Unified DB inspection tool (connection, list dbs, vector checks).
  - Usage: `python scripts/verification/inspect_db.py {check|list|vector}`
- **`check_schema.py`**: Validates schema consistency.
- **`verify_complex.py`**, **`verify_deep_dive.py`**: Advanced ingestion validation.

### `ingestion/`
- **`s3_cli.py`**: S3 utility for listing roots and fetching real manifest samples.
  - Usage: `python scripts/ingestion/s3_cli.py {roots|sample}`
- **`batch_ingest.py`**: Tool for running batch ingestion jobs.

### `ops/`
- **`manage_embeddings.py`**: Fix NULL embeddings or force full re-embedding.
  - Usage: `python scripts/ops/manage_embeddings.py {fix|force}`
- **`apply_migration.py`**: Manual migration runner.
- **`provision_*.sh`**: System provisioning scripts.

### `legacy/`
- Deprecated or one-off analysis scripts.

## 🛠 Usage
All Python scripts should be run from the repository root (e.g., `python scripts/search/search_cli.py ...`) to ensure correct import resolution.
