# `config.py` - Centralized Configuration Management

## 1. Overview

This script is the central hub for managing all configuration settings for the EmailOps application. It provides a structured and consistent way to handle environment variables, default values, and critical paths, ensuring that the application behaves predictably across different environments.

The core of this module is the `EmailOpsConfig` dataclass, which encapsulates all configuration parameters. It follows the best practice of reading values from environment variables, with sensible fallbacks to default values.

## 2. The `EmailOpsConfig` Dataclass

This dataclass defines all configuration parameters for the application.

| Parameter | Environment Variable | Default Value | Description |
|---|---|---|---|
| **Directory Settings** | | | |
| `INDEX_DIRNAME` | `INDEX_DIRNAME` | `_index` | Name of the directory to store search indexes. |
| `CHUNK_DIRNAME` | `CHUNK_DIRNAME` | `_chunks` | Name of the directory to store processed text chunks. |
| **Processing Defaults** | | | |
| `DEFAULT_CHUNK_SIZE` | `CHUNK_SIZE` | `1600` | The target size for text chunks. |
| `DEFAULT_CHUNK_OVERLAP` | `CHUNK_OVERLAP` | `200` | The overlap between consecutive text chunks. |
| `DEFAULT_BATCH_SIZE` | `EMBED_BATCH` | `64` | The number of items to process in a single batch for embeddings. |
| `DEFAULT_NUM_WORKERS` | `NUM_WORKERS` | `os.cpu_count() or 4` | Number of worker processes for parallel processing. |
| **Embedding Provider** | | | |
| `EMBED_PROVIDER` | `EMBED_PROVIDER` | `vertex` | The embedding provider to use (e.g., 'vertex'). |
| `VERTEX_EMBED_MODEL` | `VERTEX_EMBED_MODEL` | `gemini-embedding-001` | The specific Vertex AI embedding model to use. |
| **GCP Settings** | | | |
| `GCP_PROJECT` | `GCP_PROJECT` | `None` | Your Google Cloud Platform project ID. |
| `GCP_REGION` | `GCP_REGION` | `us-central1` | The default GCP region. |
| `VERTEX_LOCATION` | `VERTEX_LOCATION` | `us-central1` | The location for Vertex AI services. |
| **Paths and Credentials** | | | |
| `SECRETS_DIR` | `SECRETS_DIR` | `secrets` | Directory to store sensitive files like API keys. |
| `GOOGLE_APPLICATION_CREDENTIALS` | `GOOGLE_APPLICATION_CREDENTIALS` | `None` | Path to your GCP service account JSON file. |
| `CREDENTIAL_FILES_PRIORITY` | - | `["embed2-474114-fca38b4d2068.json", "api-agent-470921-4e2065b2ecf9.json", "apt-arcana-470409-i7-ce42b76061bf.json", "crafty-airfoil-474021-s2-34159960925b.json", "my-project-31635v-8ec357ac35b2.json", "semiotic-nexus-470620-f3-3240cfaf6036.json"]` | Prioritized list of credential filenames to search for. |
| **File Patterns** | | | |
| `ALLOWED_FILE_PATTERNS` | - | `["*.txt", "*.pdf", "*.docx", "*.doc", "*.xlsx", "*.xls", "*.md", "*.csv"]` | Glob patterns for file types that are allowed for processing. |
| **Security** | | | |
| `ALLOW_PARENT_TRAVERSAL` | `ALLOW_PARENT_TRAVERSAL` | `false` | If `true`, allows file operations outside the project directory. |
| `COMMAND_TIMEOUT_SECONDS` | `COMMAND_TIMEOUT` | `3600` | Timeout for external commands. |
| **Logging & Monitoring** | | | |
| `LOG_LEVEL` | `LOG_LEVEL` | `INFO` | The logging level for the application. |
| `ACTIVE_WINDOW_SECONDS` | `ACTIVE_WINDOW_SECONDS` | `120` | Time window for monitoring active processes. |

## 3. Core Logic and Workflows

### 3.1. Configuration Loading

The configuration is loaded using a singleton pattern. The `get_config()` function ensures that the `EmailOpsConfig` is instantiated only once.

```mermaid
graph TD
    A[Application requests config via get_config()] --> B{Is global _config instance None?};
    B -- Yes --> C[Create new EmailOpsConfig instance];
    C --> D[Load values from environment variables];
    D --> E[Set global _config instance];
    E --> F[Return instance];
    B -- No --> F;
```

### 3.2. Credential Discovery (`get_credential_file`)

This is a critical security and functionality workflow. The application automatically finds the correct Google Cloud credentials.

```mermaid
graph TD
    A[get_credential_file() is called] --> B{Is GOOGLE_APPLICATION_CREDENTIALS env var set?};
    B -- Yes --> C{Does the file exist?};
    C -- Yes --> D[Return file path];
    C -- No --> E[Continue to search];
    B -- No --> E;
    E --> F[Get secrets directory path];
    F --> G{Does secrets directory exist?};
    G -- No --> H[Return None];
    G -- Yes --> I[Iterate through CREDENTIAL_FILES_PRIORITY list];
    I --> J{Does the file exist in secrets dir?};
    J -- Yes --> K{Is it a valid JSON with project_id and client_email?};
    K -- Yes --> L[Return file path];
    K -- No --> I;
    J -- No --> I;
    I -- All files checked --> H;
```

### 3.3. Secrets Directory Resolution (`get_secrets_dir`)

The `get_secrets_dir` method follows a specific search order to find the `secrets` directory.

1.  **Absolute Path:** If `SECRETS_DIR` is an absolute path, it's used directly.
2.  **Relative to Current Directory:** It checks for a `secrets` directory relative to the current working directory.
3.  **Relative to Package Root:** It checks for a `secrets` directory relative to the main `emailops` package.
4.  **Default:** If not found, it returns the default path, even if it doesn't exist.

## 4. Key Functions

-   **`get_config() -> EmailOpsConfig`**: The main entry point to access the application's configuration. It implements the singleton pattern.
-   **`reset_config()`**: A helper function used primarily in testing to clear the global configuration object, allowing for tests to run with different configurations in isolation.
-   **`update_environment()`**: This method syncs the configuration back to the environment variables (`os.environ`). This is crucial for ensuring that any subprocesses or external tools spawned by the application inherit the correct settings. If `GCP_PROJECT` is set, it also sets `GOOGLE_CLOUD_PROJECT` and `VERTEX_PROJECT`.
-   **`to_dict() -> dict`**: Converts the configuration object to a dictionary, which is useful for logging, debugging, or serialization.

## 5. Singleton Pattern

The module uses a global variable `_config` and the `get_config()` function to ensure that there's only one configuration object throughout the application's lifecycle.

**Why is this important?**

-   **Consistency:** Every part of the application gets the exact same configuration values.
-   **Efficiency:** The environment variables are read and parsed only once.
-   **State Management:** It provides a single, predictable source of truth for configuration.

```python
# Global configuration instance
_config: EmailOpsConfig | None = None

def get_config() -> EmailOpsConfig:
    '''
    Get the global configuration instance (singleton pattern).
    '''
    global _config
    if _config is None:
        _config = EmailOpsConfig.load()
    return _config
```
