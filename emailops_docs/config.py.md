# `config.py` - Centralized Configuration Management

## 1. Overview

This module provides a centralized system for managing all configuration settings for the EmailOps application. It uses a `dataclass` to define all parameters, loads values from environment variables with sensible defaults, and provides utilities for credential discovery and environment propagation.

**Key Features:**
- **Centralized Configuration**: A single `EmailOpsConfig` dataclass holds all settings.
- **Environment-driven**: Loads configuration from environment variables.
- **Singleton Pattern**: `get_config()` ensures a single configuration instance globally.
- **Credential Auto-Discovery**: Automatically finds and validates GCP service account credentials.
- **Dynamic Environment Updates**: Propagates settings to child processes.

---

## 2. The `EmailOpsConfig` Dataclass

This is the core of the configuration system, defining all configurable parameters.

### 2.1 Dataclass Schema

```python
@dataclass
class EmailOpsConfig:
    # Directory names
    INDEX_DIRNAME: str
    CHUNK_DIRNAME: str

    # Processing defaults
    DEFAULT_CHUNK_SIZE: int
    DEFAULT_CHUNK_OVERLAP: int
    DEFAULT_BATCH_SIZE: int
    DEFAULT_NUM_WORKERS: int

    # Embedding provider settings
    EMBED_PROVIDER: str
    VERTEX_EMBED_MODEL: str

    # GCP settings
    GCP_PROJECT: str | None
    GCP_REGION: str
    VERTEX_LOCATION: str

    # Paths
    SECRETS_DIR: Path
    GOOGLE_APPLICATION_CREDENTIALS: str | None

    # File patterns
    ALLOWED_FILE_PATTERNS: list[str]

    # Credential management
    CREDENTIAL_FILES_PRIORITY: list[str]

    # Security settings
    ALLOW_PARENT_TRAVERSAL: bool
    COMMAND_TIMEOUT_SECONDS: int

    # Logging and Monitoring
    LOG_LEVEL: str
    ACTIVE_WINDOW_SECONDS: int

    # Email settings
    SENDER_LOCKED_NAME: str
    SENDER_LOCKED_EMAIL: str
    MESSAGE_ID_DOMAIN: str
```

### 2.2 Configuration Parameters

| Parameter | Environment Variable | Default | Description |
|---|---|---|---|
| `INDEX_DIRNAME` | `INDEX_DIRNAME` | `_index` | Name of the directory to store the search index. |
| `CHUNK_DIRNAME` | `CHUNK_DIRNAME` | `_chunks` | Name of the directory to store processed text chunks. |
| `DEFAULT_CHUNK_SIZE` | `CHUNK_SIZE` | `1500` | Target size of each text chunk in characters. |
| `DEFAULT_CHUNK_OVERLAP` | `CHUNK_OVERLAP` | `150` | Number of characters to overlap between chunks. |
| `DEFAULT_BATCH_SIZE` | `EMBED_BATCH` | `128` | Batch size for embedding generation. |
| `DEFAULT_NUM_WORKERS` | `NUM_WORKERS` | CPU count (or 4) | Default number of parallel workers for processing. |
| `EMBED_PROVIDER` | `EMBED_PROVIDER` | `vertex` | The embedding provider to use (e.g., 'vertex'). |
| `VERTEX_EMBED_MODEL` | `VERTEX_EMBED_MODEL` | `gemini-embedding-001` | The specific Vertex AI embedding model. |
| `GCP_PROJECT` | `GCP_PROJECT` | `None` | Google Cloud project ID. |
| `GCP_REGION` | `GCP_REGION` | `us-central1` | Google Cloud region. |
| `VERTEX_LOCATION` | `VERTEX_LOCATION` | `us-central1` | Location for Vertex AI resources. |
| `SECRETS_DIR` | `SECRETS_DIR` | `/secrets` | Directory to search for credential files. |
| `GOOGLE_APPLICATION_CREDENTIALS` | `GOOGLE_APPLICATION_CREDENTIALS` | `None` | Path to the GCP service account JSON file. |
| `ALLOWED_FILE_PATTERNS` | - | `["*.txt", "*.pdf", ...]` | Glob patterns for allowed attachment file types. |
| `CREDENTIAL_FILES_PRIORITY` | - | `[]` | Sorted list of credential files found in `SECRETS_DIR`. |
| `ALLOW_PARENT_TRAVERSAL` | `ALLOW_PARENT_TRAVERSAL` | `False` | If `True`, allows '..' in file paths (security risk). |
| `COMMAND_TIMEOUT_SECONDS` | `COMMAND_TIMEOUT` | `3600` | Timeout for external command execution. |
| `LOG_LEVEL` | `LOG_LEVEL` | `INFO` | Logging level for the application. |
| `ACTIVE_WINDOW_SECONDS` | `ACTIVE_WINDOW_SECONDS` | `120` | Time window for monitoring active operations. |
| `SENDER_LOCKED_NAME` | `SENDER_LOCKED_NAME` | `""` | Default sender name for outgoing emails. |
| `SENDER_LOCKED_EMAIL` | `SENDER_LOCKED_EMAIL` | `""` | Default sender email address. |
| `MESSAGE_ID_DOMAIN` | `MESSAGE_ID_DOMAIN` | `""` | Domain used for generating unique Message-IDs. |

---

## 3. Core Functions

### 3.1 Configuration Loading

#### `get_config() -> EmailOpsConfig`
This function implements the singleton pattern. It returns the global `EmailOpsConfig` instance, creating it if it doesn't exist.

**Workflow:**
```mermaid
graph TD
    A[Call get_config()] --> B{Is global _config None?};
    B -- Yes --> C[EmailOpsConfig.load() is called];
    C --> D[New config instance created];
    D --> E[Set as global _config];
    E --> F[Return instance];
    B -- No --> F;
```

#### `EmailOpsConfig.load() -> EmailOpsConfig`
A class method that creates a new instance of `EmailOpsConfig`, loading all values from environment variables or their defaults.

### 3.2 Credential Discovery and Validation

This is a critical feature for simplifying deployment. The system automatically finds and validates GCP credentials.

#### `_is_valid_service_account_json(p: Path) -> bool`
A static method that performs strict validation on a JSON file to ensure it is a valid GCP service account key.

**Validation Checks:**
1.  **Structure**: Must be a dictionary containing required keys (`type`, `project_id`, `private_key`, etc.).
2.  **Type**: The `type` field must be `"service_account"`.
3.  **Private Key**: Must start and end with the correct PEM block markers.
4.  **Key ID**: Must be a hex string of a minimum length.
5.  **Email Format**: Must be a valid service account email format (e.g., `...@...iam.gserviceaccount.com`).
6.  **Project ID**: Must meet GCP's length and character requirements.
7.  **Token Validity** (if `google-auth` is installed):
    - Attempts to create a credentials object.
    - Checks if the credentials have expired.

#### `get_credential_file() -> Path | None`
This method finds a valid credential file by searching in a specific order of priority.

**Search Order:**
1.  **Environment Variable**: Checks if `GOOGLE_APPLICATION_CREDENTIALS` is set and points to a valid file.
2.  **Secrets Directory**: Searches the `SECRETS_DIR` for JSON files, sorted by the `CREDENTIAL_FILES_PRIORITY` list. The first valid file found is returned.

### 3.3 Environment Propagation

#### `update_environment() -> None`
This method updates the current process's environment variables (`os.environ`) with the values from the `EmailOpsConfig` instance.

**Purpose:**
- Ensures that child processes (e.g., subprocesses, parallel workers) inherit the correct configuration.
- Derives the `GCP_PROJECT` from the selected service account file if it's not already set.

---

## 4. Usage

The configuration is accessed throughout the application via the `get_config()` function.

```python
from emailops.config import get_config

# Get the global configuration instance
config = get_config()

# Access configuration values
chunk_size = config.DEFAULT_CHUNK_SIZE
gcp_project = config.GCP_PROJECT

# The configuration is automatically loaded on first call
# and reused on subsequent calls.