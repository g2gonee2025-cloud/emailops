# `doctor.py` - System Diagnostics and Setup

## 1. Overview

`doctor.py` is a command-line utility designed to diagnose the health of the EmailOps environment. It acts as a "first aid" tool to verify dependencies, check the integrity of the search index, and test core functionalities like embeddings. It can even automatically install missing packages.

This tool is essential for ensuring a smooth and error-free experience when using the EmailOps application.

## 2. Dependencies

This module depends on:
- **`index_metadata` module**: Used for reading index mappings and loading index metadata
- **External packages**: Various embedding providers and document extractors (see Provider-Specific Dependencies section)

## 3. Core Workflows

The doctor script performs three main tasks, which can be run together or separately.

### 3.1. Dependency Check & Installation

This is the primary workflow. It intelligently checks for required and optional Python packages based on the configured embedding provider.

```mermaid
graph TD
    A[Start doctor.py] --> B[Get configured provider (e.g., 'vertex')];
    B --> C[Identify critical & optional packages for this provider];
    C --> D{Check for missing critical packages};
    D -- Missing --> E{auto-install enabled?};
    E -- Yes --> F[Attempt 'pip install' for missing criticals];
    F --> G{Install successful?};
    G -- Yes --> H[Proceed];
    G -- No --> I[Log error and exit];
    E -- No --> J[Log error with manual install instructions];
    D -- All Present --> H;
    H --> K{Check for missing optional packages};
    K -- Missing --> L{auto-install enabled?};
    L -- Yes --> M[Attempt 'pip install' for missing optionals];
    L -- No --> N[Log warning with manual install instructions];
    K -- All Present --> O[All dependency checks complete];
    M --> O;
    N --> O;
    J --> O;
```

### 3.2. Index Health Check (`--check-index`)

If the `--check-index` flag is used, the script inspects the search index directory (`_index`) using functions from the `index_metadata` module.

```mermaid
graph TD
    A[Start Index Check] --> B{Does _index directory exist?};
    B -- No --> C[Skip check];
    B -- Yes --> D[Load index metadata via index_metadata.load_index_metadata()];
    D --> E[Load mapping via index_metadata.read_mapping()];
    E --> F[Calculate statistics (e.g., num documents, num conversations)];
    F --> G[Display statistics];
    G --> H[Check for provider compatibility];
    H --> I{Is index provider same as current provider?};
    I -- Yes --> J[Log 'Compatibility: OK'];
    I -- No --> K[Log 'Compatibility: FAIL' with warning];
```

### 3.3. Embedding Probe (`--check-embeddings`)

If the `--check-embeddings` flag is used, the script performs a live test of the embedding service.

```mermaid
graph TD
    A[Start Embedding Probe] --> B[Import embed_texts function from llm_client];
    B --> C[Send a sample text ("test") to the embedding API];
    C --> D{Did the API call succeed?};
    D -- Yes --> E[Get embedding dimension from the result];
    E --> F[Log 'Success' with dimension];
    D -- No --> G[Log 'Failed' with error message];
```

## 4. Key Logic Details

### Provider-Specific Dependencies

The script is smart about dependencies. It knows which packages are needed for each provider, reducing unnecessary installations. In addition to the provider-specific optional packages, a set of common packages are always considered optional (`numpy`, `faiss-cpu`, `pypdf`, `python-docx`, `pandas`, `openpyxl`).

| Provider | Critical Packages | Optional Packages |
|---|---|---|
| `vertex` | `google-genai`, `google-cloud-aiplatform` | `google-auth`, `google-cloud-storage` |
| `openai` | `openai` | `tiktoken` |
| `azure` | `openai` | `azure-identity` |
| `cohere` | `cohere` | - |
| `huggingface` | `huggingface_hub` | - |
| `qwen` | `requests` | - |
| `local` | `sentence-transformers` | `torch`, `transformers` |

### Safe Import Checking

It uses `importlib.import_module` inside a `try...except` block to safely check if a library is installed without terminating the script if it's not found.

To handle cases where the package name installed by `pip` differs from the name used for import (e.g., `faiss-cpu` is imported as `faiss`), the script uses a mapping dictionary called `_PKG_IMPORT_MAP`. This ensures that the check is accurate.

### Helper Functions

The module includes several internal helper functions not exposed in the CLI:

- **`_load_mapping(index_dir: Path)`**: Loads the index mapping from the specified directory using `index_metadata.read_mapping()`
- **`_get_index_statistics(index_dir: Path)`**: Calculates statistics about the index including document count, conversation count, and total characters
- **`_summarize_index_compat(index_dir: Path, provider: str)`**: Checks if the index was built with the same provider currently configured
- **`_probe_embeddings(provider: str)`**: Tests the embedding functionality by generating a test embedding

## 5. CLI Usage

You can run the doctor script from your terminal.

**Command-line Arguments:**

| Argument | Description | Default |
|---|---|---|
| `--root` | The project root directory. | `.` |
| `--provider` | The embedding provider to check against. | `vertex` |
| `--auto-install` | If set, automatically `pip install` missing packages. | `False` |
| `--check-index` | If set, runs the index health check. | `False` |
| `--check-embeddings` | If set, runs the embedding probe. | `False` |

**Examples:**

```bash
# Run a full check for the vertex provider and auto-install packages
python -m emailops.doctor --provider vertex --auto-install --check-index --check-embeddings

# Check for openai dependencies without installing
python -m emailops.doctor --provider openai

# Only check the index health
python -m emailops.doctor --check-index
```

## 6. Constants and Configuration

The module defines several constants:

- **`INDEX_DIRNAME_DEFAULT`**: Default index directory name from environment variable or `"_index"`
- **`REPO_ROOT`**: Path to the repository root directory (parent of the emailops package)
- **`_PKG_IMPORT_MAP`**: Dictionary mapping package names to their import names

## 7. Developer Notes

- The module uses a library-safe logger: `logging.getLogger("emailops.doctor")`
- Timeouts for package installation are set to 300 seconds
- The module gracefully handles missing `index_metadata` module imports
- All functions that interact with the index handle exceptions gracefully to prevent crashes