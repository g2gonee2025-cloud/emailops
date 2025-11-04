# `emailops.core_config_models`

**Primary Goal:** To define the structure of the application's configuration using Python `dataclasses`. This file acts as a schema, ensuring that all configuration settings are strongly typed and organized logically into distinct categories.

## Directory Mapping

```
.
└── emailops/
    ├── core_config.py
    └── core_config_models.py
```

---

## Core Functions & Connections

This module contains no executable logic; it is purely declarative. It defines the data structures that are instantiated and managed by `emailops.core_config.py`.

### `_env(key, default, value_type)`

- **Purpose:** A private helper function used as a factory for `dataclass` fields. It provides a clean way to initialize a field's value from an environment variable, falling back to a specified default if the variable is not set.
- **Functionality:**
    1.  Checks for an environment variable with a prefixed name (e.g., `EMAILOPS_LOG_LEVEL`) and falls back to the direct name (`LOG_LEVEL`).
    2.  If an environment variable is found, it attempts to cast the value to the specified `value_type` (e.g., `int`, `bool`, `float`).
    3.  If the variable is not found or a casting error occurs, it returns the `default` value.
- **Connections:** This function is used extensively within the dataclasses below to make the configuration system "environment-aware" out of the box.

---

## Configuration Dataclasses

Each dataclass represents a logical grouping of related settings.

### `DirectoryConfig`
- **Purpose:** Defines paths for key directories used by the application.
- **Attributes:**
    - `index_dirname`: The name of the subdirectory where the vector index is stored.
    - `chunk_dirname`: The name of the subdirectory for storing processed text chunks.
    - `secrets_dir`: The path to the directory containing credentials and other secrets.

### `CoreConfig`
- **Purpose:** Holds the most fundamental, top-level settings.
- **Attributes:**
    - `export_root`: The root directory of the exported email data.
    - `provider`: The default embedding and LLM provider (e.g., "vertex", "openai").
    - `persona`: The default persona to be used by the LLM when drafting emails.

### `ProcessingConfig`
- **Purpose:** Governs the parameters for text processing and embedding.
- **Attributes:**
    - `chunk_size`, `chunk_overlap`: Parameters for splitting documents into smaller pieces for the language model.
    - `batch_size`: The number of items to process in a single batch during embedding.
    - `num_workers`: The number of parallel processes to use for data processing.

### `EmbeddingConfig`
- **Purpose:** Specifies the models to be used for generating embeddings and responses, particularly for Google Vertex AI.
- **Attributes:**
    - `vertex_embed_model`: The specific model for creating vector embeddings.
    - `vertex_model`: The generative model for tasks like chatting and drafting.
    - `vertex_embed_dim`, `vertex_output_dim`: The dimensions of the embedding vectors, which can be important for some vector databases.

### `GcpConfig`
- **Purpose:** Contains settings specific to Google Cloud Platform.
- **Attributes:**
    - `gcp_region`, `vertex_location`, `gcp_project`: Identifiers for the GCP project and the location of Vertex AI services.

### `RetryConfig`
- **Purpose:** Defines a comprehensive strategy for handling transient network errors or API failures.
- **Attributes:**
    - `max_retries`, `initial_backoff_seconds`, `backoff_multiplier`: Parameters for implementing exponential backoff.
    - `rate_limit_per_sec`, `rate_limit_capacity`: Settings for a token bucket rate limiter.
    - `circuit_failure_threshold`, `circuit_reset_seconds`: Parameters for a circuit breaker pattern, which stops sending requests for a period after a certain number of consecutive failures.

### `SearchConfig`
- **Purpose:** A detailed set of parameters that control the behavior of the search and retrieval system.
- **Attributes:**
    - `half_life_days`, `recency_boost_strength`: Control how much more recent documents are favored in search results.
    - `sim_threshold`: The minimum similarity score for a document to be considered relevant.
    - `rerank_alpha`, `mmr_lambda`: Parameters that tune the hybrid search (RRF) and diversification (MMR) algorithms.
    - `k`: The number of search results to retrieve.

### `EmailConfig`
- **Purpose:** Defines rules and default values for email generation.
- **Attributes:**
    - `reply_policy`: The default behavior for replying (e.g., `reply_all`).
    - `sender_locked_name`, `sender_locked_email`: A fixed sender identity.
    - `allowed_senders`: A set of email addresses that are permitted to be used as the sender.

### `SummarizerConfig`, `LimitsConfig`, `SystemConfig`, `SecurityConfig`
- **Purpose:** These classes group together various other settings related to summarization, system-level limits (e.g., max file size), logging, and security flags.

### `SensitiveConfig`
- **Purpose:** This class is specifically for holding API keys and other secrets. Its fields default to `None` and are expected to be populated exclusively from environment variables or a secure vault, not checked into version control.

### `FilePatternsConfig`
- **Purpose:** Defines which file types the system is allowed to process.
- **Attributes:**
    - `allowed_file_patterns`: A list of glob patterns (e.g., `*.pdf`, `*.docx`) for whitelisting files during ingestion.

### `UnifiedConfig`
- **Purpose:** This class holds miscellaneous settings that might be shared across different components or used for UI state.

---

## How It Connects

The dataclasses in this file serve as the "schema" for the `EmailOpsConfig` class in `emailops/core_config.py`. The `EmailOpsConfig` class has an attribute for each of these models (e.g., `self.directories: DirectoryConfig`), creating a nested, organized structure that is easy to navigate both in code (`config.directories.index_dirname`) and in the resulting JSON file. This strong typing and clear organization make the configuration system robust and easy to maintain.