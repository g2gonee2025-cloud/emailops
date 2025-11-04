# `emailops.tool_doctor`

**Primary Goal:** To act as a command-line diagnostic and setup verification tool for the EmailOps application. It provides a suite of checks to ensure that the environment is correctly configured, dependencies are installed, and core components like the search index and embedding providers are functional.

## Directory Mapping

```
.
└── emailops/
    └── tool_doctor.py
```

---

## Core Functions & Connections

The `main()` function orchestrates a series of checks based on the command-line arguments provided by the user.

### Dependency Management

- **`check_and_install_dependencies(...)`:**
    - **Purpose:** This is the primary function for managing Python package dependencies.
    - **Functionality:**
        1.  It determines the required (`critical`) and `optional` packages for a given LLM provider by calling `_packages_for_provider()`.
        2.  It checks for the presence and importability of each package using `_try_import()`. This helper is robust, distinguishing between packages that are "not_installed" and those that are "broken" (installed but raise an error on import).
        3.  If the `--auto-install` flag is used, it will attempt to install any missing packages by calling `_install_packages()`, which safely runs `pip install` in a subprocess with a timeout.
- **Connections:** This is a crucial setup step. The rest of the application cannot function if its core dependencies are missing. This tool provides a user-friendly way to diagnose and fix these issues.

### Index Health Checks

- **`_get_index_statistics(...)`:**
    - **Purpose:** To provide a high-level overview of the contents of an existing index.
    - **Functionality:** It reads the `mapping.json` file and calculates key metrics like the total number of documents, the number of unique conversations, and the total number of characters indexed.
    - **Connections:** It depends on `indexing_metadata.read_mapping` to load the index manifest.
- **`_summarize_index_compat(...)`:**
    - **Purpose:** To check for a common and critical configuration error: trying to use an index with a different LLM provider than the one it was built with.
    - **Functionality:** It reads the `meta.json` file (via `load_index_metadata`) to find out which provider was used to create the index and compares it to the provider currently being used. Mismatched embedding models produce incompatible vectors, so this check is vital for preventing nonsensical search results.

### Live Provider Probing

- **`_probe_embeddings(...)`:**
    - **Purpose:** To perform a live, end-to-end test of the embedding functionality for a given provider.
    - **Functionality:** It attempts to embed a simple test string ("test") by calling `llm_client_shim.embed_texts`. If the call succeeds, it confirms that the connection to the provider is working, the API keys are valid, and it reports the dimension of the returned vector.
    - **Connections:** This is a powerful diagnostic because it directly invokes the core `embed_texts` function, testing the full stack from the client shim down to the `llm_runtime` and the external API.

---

## Command-Line Interface (`main`)

The `main` function parses command-line arguments and runs the selected checks, reporting the results in either a human-readable format or as structured JSON.

- **`--check-dependencies`:** Runs the dependency check for the specified `--provider`.
- **`--check-index`:** Runs the index statistics and compatibility checks.
- **`--check-embeddings`:** Runs the live embedding probe.
- **`--auto-install`:** Enables automatic installation of missing dependencies.
- **`--json`:** Switches the output to a machine-readable JSON format, which is useful for scripting or integration with other tools.

The tool uses system exit codes to signal the outcome, making it suitable for use in automated CI/CD pipelines:
- `0`: All checks passed.
- `2`: A critical dependency is missing.
- `3`: An issue was found with the index.
- `4`: The embedding probe failed.

## Key Design Patterns

- **Diagnostic Tool:** The entire module is a classic example of a diagnostic tool, designed to help developers and administrators quickly identify and resolve common configuration and setup issues.
- **Separation of Concerns:** The module is well-structured, with separate functions for each distinct check (dependencies, index, embeddings).
- **Graceful Fallbacks:** The dependency checking (`_try_import`) and provider probing (`_probe_embeddings`) are wrapped in `try...except` blocks, so a failure in one check does not prevent the others from running.
- **User-Friendly Reporting:** The tool provides clear, actionable feedback in its human-readable output, telling the user *what* is wrong and often *how* to fix it (e.g., "Run 'pip install ...'").