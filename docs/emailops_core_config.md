# `emailops.core_config`

**Primary Goal:** To provide a centralized, thread-safe, and robust system for managing all configuration settings for the EmailOps application. It consolidates settings from environment variables and JSON files into a single, easily accessible object.

## Directory Mapping

```
.
└── emailops/
    ├── core_config.py
    └── core_config_models.py
```

---

## Core Functions & Connections

This module is the single source of truth for all configuration parameters. It is designed to be imported by any other module in the application that needs access to settings.

### `get_config()`

- **Purpose:** This is the most critical function in the module. It returns a global, thread-safe singleton instance of the `EmailOpsConfig` class. The first time it's called, it loads the configuration; subsequent calls return the already-loaded instance.
- **Design Pattern (Singleton with Double-Checked Locking):**
    1.  It first checks if the global `_config` object exists. If it does, it returns it immediately without acquiring a lock, which is very fast.
    2.  If `_config` is `None`, it acquires a thread lock (`_config_lock`).
    3.  It checks *again* if `_config` is `None` (in case another thread initialized it while the current thread was waiting for the lock).
    4.  If it's still `None`, it calls `EmailOpsConfig.load()` to create the instance.
- **Connections:** This function is called by virtually every other module in the application that needs to know about file paths, API keys, model names, or any other setting. For example, `emailops.cli` calls it to set up the environment before running a command.

### `EmailOpsConfig` Class

- **Purpose:** A dataclass that acts as a container for all other configuration models (e.g., `DirectoryConfig`, `GcpConfig`, `SecurityConfig`). It aggregates all settings into a single, structured object.
- **Connections:**
    - **Imports:** It is built upon the dataclasses defined in `emailops.core_config_models`. This separation keeps the model definitions clean and separate from the loading/management logic.
    - **`load()` classmethod:** Handles the logic of loading configuration from a JSON file and overriding values with any environment variables (e.g., `EMAILOPS_*`). It also includes robust error handling for corrupt JSON files.
    - **`save()` method:** Serializes the current configuration state to a JSON file, ensuring that Path objects are correctly converted to strings.
    - **`update_environment()` method:** This is a crucial method for ensuring that child processes (like the indexer launched from `cli.py`) inherit the correct settings. It takes the values from the config object and sets them as environment variables.
    - **`discover_credential_files()` and `get_credential_file()`:** These methods provide a robust way to automatically find and validate GCP service account credentials stored in the `secrets` directory. This avoids hardcoding credential paths and adds a layer of validation.

### `reset_config()`

- **Purpose:** A utility function, primarily for testing, to reset the global configuration singleton. This allows tests to run in isolation with different configurations.
- **Connections:** Used by testing frameworks to ensure that each test starts with a clean configuration slate.

---

## Configuration Loading Logic

The system employs a layered approach to loading configuration, providing flexibility for different environments:

1.  **Environment Variables (`.env`):** The `dotenv` library loads key-value pairs from a `.env` file into the environment. This is the first layer.
2.  **JSON File (optional):** The `EmailOpsConfig.load(path)` method can load a baseline configuration from a JSON file.
3.  **Environment Variable Overrides:** After loading from JSON, the `load` method checks for environment variables (e.g., `EMAILOPS_GCP_PROJECT`) that can override the values from the file.
4.  **`get_config()` Singleton:** This function orchestrates the loading process and ensures it only happens once.

This hierarchy allows developers to have a default `emailops_config.json`, but allows system administrators or CI/CD pipelines to override specific settings using environment variables without modifying the file.

---

## Key Design Patterns

- **Singleton:** The `get_config()` function ensures that there is only one instance of the `EmailOpsConfig` object throughout the application's lifecycle, preventing configuration drift.
- **Thread Safety:** The use of `threading.RLock` in `get_config()` and `reset_config()` prevents race conditions when multiple threads try to access or modify the global configuration object simultaneously.
- **Facade Pattern:** The `EmailOpsConfig` class acts as a facade, providing a simple, unified interface to a complex subsystem of different configuration categories.
- **Separation of Concerns:** The logic for loading and managing the configuration (`core_config.py`) is kept separate from the data structures that hold the configuration (`core_config_models.py`).