# `emailops.core_exceptions`

**Primary Goal:** To establish a clear, hierarchical, and context-rich set of custom exceptions for the EmailOps application. This allows for more precise error handling and debugging than relying on generic built-in exceptions like `ValueError` or `Exception`.

## Directory Mapping

```
.
└── emailops/
    └── core_exceptions.py
```

---

## Core Exception: `EmailOpsError`

This is the cornerstone of the exception hierarchy.

- **Purpose:** Serves as the base class for all custom exceptions within the application. Any code that wants to catch a general, known application error can simply `except EmailOpsError:`.
- **Key Features:**
    - **`message`**: A human-readable description of what went wrong.
    - **`error_code`**: An optional, machine-readable string (e.g., "INDEX_CORRUPT", "PROVIDER_AUTH_FAILURE"). This is invaluable for programmatic error handling, allowing the application to take specific actions based on the type of error without having to parse the message string.
    - **`context`**: A dictionary to hold any additional, relevant data for debugging (e.g., the file path that caused an error, the arguments to a failed API call).
    - **`to_dict()` method:** A utility to serialize the exception's data, making it easy to log in a structured format (like JSON).

---

## Specific Exception Subclasses

Each subclass of `EmailOpsError` represents a distinct category of problem, allowing for more granular `try...except` blocks.

### `ConfigurationError`
- **Purpose:** Raised when there is a problem with the application's configuration, such as a missing setting in `.env` or an invalid value in `emailops_config.json`.
- **Connections:** Primarily raised by `emailops.core_config` during the loading and validation of settings.

### `EmailOpsIndexError`
- **Purpose:** Used for all errors related to the search index (e.g., the index directory is not found, the index is corrupt, or it's incompatible with the current version).
- **Note:** It was specifically renamed from `IndexError` to avoid conflict with Python's built-in `IndexError`, a good practice for clarity.

### `EmbeddingError`
- **Purpose:** Raised during the vector embedding process. This could be due to issues with the embedding provider (like an API outage) or data problems (like a dimension mismatch).
- **Key Feature (`retryable`):** This boolean flag is critical. It allows the calling code to distinguish between a temporary network glitch that might be resolved by retrying (`retryable=True`) and a permanent error like an invalid API key that should not be retried (`retryable=False`).

### `ProcessingError`
- **Purpose:** A general exception for failures that occur during the processing of documents or text, such as a failure in parsing or cleaning.

### `ValidationError`
- **Purpose:** Used when input data fails a validation check.
- **Key Features (`field`, `rule`):** This exception can store which specific `field` failed validation and the `rule` it violated. This is extremely useful for providing precise feedback to a user or in an API response.

### `ProviderError`
- **Purpose:** A specific error for when an external, third-party service (like an LLM or embedding API provider) fails.
- **Key Features (`provider`, `retryable`):** It stores which `provider` failed and whether the operation is `retryable`, allowing for sophisticated error handling (e.g., "The 'openai' provider failed with a rate limit error, which is retryable.").

### `FileOperationError`
- **Purpose:** Raised for any failures related to file system I/O (reading, writing, deleting).
- **Key Features (`file_path`, `operation`):** Storing the `file_path` and the `operation` ("read", "write") that failed makes debugging file permission or disk space issues much more straightforward.

### `TransactionError` & `SecurityError`
- **Purpose:** These are more specialized exceptions for handling failures in transactional operations (e.g., during an atomic index update) and for security-related events (e.g., detecting a potential injection attack).

---

## Backward Compatibility & Aliases

The file includes aliases like `ProcessorError = ProcessingError`. This is a thoughtful touch for maintaining backward compatibility. If other parts of the code were previously using `ProcessorError`, they won't break after the exception was renamed or refactored to `ProcessingError`.

---

## How It Connects & Key Design Patterns

- **Hierarchical Exception Handling:** The class hierarchy (`ProviderError` inherits from `EmailOpsError`) allows for flexible error handling. A developer can write a narrow `except ProviderError:` block to handle a specific API failure, or a broad `except EmailOpsError:` block to catch any known application error, or even a final `except Exception:` for unexpected problems.
- **Context Preservation:** By adding attributes like `error_code`, `context`, `field`, and `file_path`, these exceptions carry much more information than a simple message string. This is a core principle of robust software design, as it separates the error *description* from the error *data*, enabling better logging, monitoring, and automated responses.
- **Fail Fast & Specific:** The existence of these specific exceptions encourages a "fail fast" approach. Instead of catching a generic exception and trying to figure out what happened, a function can raise a `FileOperationError` immediately, providing clear and immediate information about the root cause of the failure.