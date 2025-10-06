# 📂 Data Management Guide for EmailOps Vertex AI

## Overview

This guide explains how to properly manage runtime data for the EmailOps Vertex AI project. All runtime data (chunks, indexes, logs, metadata) should be stored OUTSIDE the repository to keep it clean and focused on code only.

## ⚠️ Important: No Runtime Data in Repository

The following types of files should NEVER be committed to the repository:
- Index files (`.faiss`, `.pkl`, `*_index/`)
- Chunk data (`*_chunks/`, `chunk_*`)
- Metadata files (`*metadata*`, `*.meta`)
- Log files (`*.log`, `logs/`)
- Database files (`*.db`, `*.sqlite`)
- Cache directories (`__pycache__/`, `.cache/`)
- Sample/test data (`sample/`, `test_output/`)
- Environment directories (`.conda/`, `.venv/`, `env/`)

## 📁 Recommended Directory Structure

Set up your data directories OUTSIDE the repository:

```
C:/EmailOpsData/                    # Or any location outside the repo
├── indexes/                        # All index files
│   ├── production/                 # Production indexes
│   │   ├── vertex_index.faiss
│   │   └── metadata.pkl
│   └── development/                # Development/test indexes
│       └── test_index.faiss
├── chunks/                         # Chunked data
│   ├── batch_001/
│   ├── batch_002/
│   └── ...
├── logs/                          # All log files
│   ├── indexing/
│   ├── processing/
│   └── errors/
├── cache/                         # Temporary cache data
└── backups/                       # Backup data

emailops_vertex_ai/                # Your repository (CODE ONLY)
├── cli.py
├── processing/
├── analysis/
├── diagnostics/
├── tests/
├── setup/
├── utils/
├── ui/
├── data/                          # Only configuration JSONs
│   ├── validated_accounts.json   # ✓ Config file (keep)
│   ├── account_diagnostics.json  # ✓ Config file (keep)
│   └── live_api_test_results.json # ✓ Config file (keep)
├── docs/
└── emailops/
```

## 🔧 Configuration

### 1. Environment Variables

Create a `.env` file (already in .gitignore) to specify data locations:

```bash
# Data directories (OUTSIDE repository)
INDEX_DIR=C:/EmailOpsData/indexes/production
CHUNK_DIR=C:/EmailOpsData/chunks
LOG_DIR=C:/EmailOpsData/logs
CACHE_DIR=C:/EmailOpsData/cache
METADATA_DIR=C:/EmailOpsData/metadata

# Or use relative paths from a parent directory
# INDEX_DIR=../../EmailOpsData/indexes
# CHUNK_DIR=../../EmailOpsData/chunks

# Google Cloud settings
GOOGLE_CLOUD_PROJECT=your-project-id
VERTEX_AI_LOCATION=us-central1
```

### 2. Update Your Scripts

Modify your Python scripts to use environment variables:

```python
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get data directories from environment
INDEX_DIR = Path(os.getenv('INDEX_DIR', '../EmailOpsData/indexes'))
CHUNK_DIR = Path(os.getenv('CHUNK_DIR', '../EmailOpsData/chunks'))
LOG_DIR = Path(os.getenv('LOG_DIR', '../EmailOpsData/logs'))

# Create directories if they don't exist
INDEX_DIR.mkdir(parents=True, exist_ok=True)
CHUNK_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Use in your code
index_path = INDEX_DIR / 'vertex_index.faiss'
log_file = LOG_DIR / f'indexing_{datetime.now():%Y%m%d_%H%M%S}.log'
```

## 🚀 Running the Application

### Local Development

```bash
# Set up data directories (one time only)
mkdir C:\EmailOpsData
mkdir C:\EmailOpsData\indexes
mkdir C:\EmailOpsData\chunks
mkdir C:\EmailOpsData\logs

# Configure environment
copy .env.example .env
# Edit .env with your data paths

# Run commands
python cli.py index
python cli.py analyze --files
```

### Docker Usage

Mount external data directories as volumes:

```bash
docker run -v C:/EmailOpsData:/data \
           -v $(pwd)/.env:/app/.env:ro \
           emailops-vertex-ai:latest \
           python cli.py index
```

Or with docker-compose.yml:

```yaml
version: '3.8'
services:
  emailops:
    image: emailops-vertex-ai:latest
    volumes:
      - C:/EmailOpsData:/data  # External data directory
      - ./.env:/app/.env:ro
    environment:
      - INDEX_DIR=/data/indexes
      - CHUNK_DIR=/data/chunks
      - LOG_DIR=/data/logs
```

## 🧹 Cleanup Commands

If any runtime data accidentally gets into the repository:

### Windows PowerShell
```powershell
# Remove all Python cache
Get-ChildItem -Path . -Include __pycache__ -Recurse -Force | Remove-Item -Force -Recurse

# Remove all log files
Get-ChildItem -Path . -Include *.log -Recurse -Force | Remove-Item -Force

# Remove all index/chunk data
Get-ChildItem -Path . -Include *_index, *_chunks, *.faiss, *.pkl -Recurse -Force | Remove-Item -Force -Recurse
```

### Linux/Mac
```bash
# Remove all Python cache
find . -type d -name __pycache__ -exec rm -rf {} +

# Remove all log files
find . -name "*.log" -type f -delete

# Remove all index/chunk data
find . -type d \( -name "*_index" -o -name "*_chunks" \) -exec rm -rf {} +
find . -type f \( -name "*.faiss" -o -name "*.pkl" \) -delete
```

## 📋 Checklist Before Committing

Before committing any changes:

- [ ] No `__pycache__` directories
- [ ] No `.log` files
- [ ] No `*_index/` or `*_chunks/` directories
- [ ] No `.faiss` or `.pkl` files
- [ ] No `.db` or `.sqlite` files
- [ ] No `sample/` or test data directories
- [ ] No `.conda/` or virtual environment directories
- [ ] No credentials or secret files
- [ ] `.gitignore` is properly configured

## 🔍 Verify Clean Repository

Run this command to check for unwanted files:

```bash
# Windows PowerShell
Get-ChildItem -Path . -Recurse -Include "*chunk*", "*index*", "*.log", "*metadata*", "*.faiss", "*.pkl", "*.db", "__pycache__", ".conda", "sample" | Where-Object { $_.PSIsContainer -or $_.Extension -ne ".py" }

# Linux/Mac
find . -type f \( -name "*.log" -o -name "*.faiss" -o -name "*.pkl" -o -name "*.db" \) -o -type d \( -name "__pycache__" -o -name "*_index" -o -name "*_chunks" \)
```

If this returns any results, clean them up before committing.

## 💾 Backup Strategy

Since runtime data is not in version control, implement a backup strategy:

1. **Regular Backups**: Schedule daily/weekly backups of your data directories
2. **Cloud Storage**: Use Google Cloud Storage to backup important indexes
3. **Version Naming**: Use timestamps in backup names: `index_20241006_production.faiss`
4. **Retention Policy**: Keep last 7 daily, 4 weekly, and 12 monthly backups

Example backup script:
```bash
# backup_data.ps1
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$backupDir = "C:\EmailOpsData\backups\$timestamp"

New-Item -ItemType Directory -Path $backupDir
Copy-Item -Path "C:\EmailOpsData\indexes" -Destination $backupDir -Recurse
Copy-Item -Path "C:\EmailOpsData\metadata" -Destination $backupDir -Recurse

# Optional: Upload to cloud
gsutil -m rsync -r $backupDir gs://your-bucket/backups/$timestamp/
```

## 📝 Summary

- **Repository**: Code only, no runtime data
- **Runtime Data**: Store in external directories (C:/EmailOpsData or similar)
- **Configuration**: Use environment variables to specify data locations
- **Docker**: Mount external directories as volumes
- **Backup**: Implement regular backups for production data
- **Clean**: Regular cleanup to ensure no runtime data in repo

Following these guidelines ensures:
- Clean, fast repository operations
- No accidental commits of large files
- Separation of code and data
- Easy deployment and scaling
- Professional development practices