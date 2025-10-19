# `processing_utils.py` - Processing Utilities

## 1. Overview

This module provides a collection of utilities to support various data processing tasks within the EmailOps application. It includes tools for batch processing, performance monitoring, and centralized configuration for document processing.

**Key Features:**
- **Batch Processing**: A `BatchProcessor` class for processing items in parallel.
- **Performance Monitoring**: A decorator to monitor the execution time of functions.
- **Centralized Configuration**: A `ProcessingConfig` dataclass for managing document processing settings.
- **Text Preprocessing**: A `TextPreprocessor` class for cleaning and preparing text for indexing.

---

## 2. Core Components

### 2.1. `ProcessingConfig` Dataclass

This dataclass centralizes configuration settings related to document processing, loading values from environment variables with sensible defaults.

**Schema:**
```python
@dataclass
class ProcessingConfig:
    max_attachment_chars: int
    excel_max_cells: int
    skip_attachment_over_mb: float
    max_total_attachment_text: int
```

**Parameters:**
| Parameter | Environment Variable | Default | Description |
|---|---|---|---|
| `max_attachment_chars` | `MAX_ATTACHMENT_TEXT_CHARS` | `500000` | Maximum characters to extract from an attachment. |
| `excel_max_cells` | `EXCEL_MAX_CELLS` | `200000` | Maximum number of cells to process from an Excel file. |
| `skip_attachment_over_mb` | `SKIP_ATTACHMENT_OVER_MB` | `0` | Skip attachments larger than this size in megabytes (0 means no limit). |
| `max_total_attachment_text`| - | `10000` | Maximum total characters of attachment text to include in the conversation. |

### 2.2. `get_processing_config()`

This function returns a singleton instance of the `ProcessingConfig`, ensuring that the configuration is loaded only once.

### 2.3. `Person` Dataclass

A dataclass to represent a person with a name and birthdate. It includes a property to calculate the person's age. This class is currently noted as unused in the codebase but is kept for backward compatibility.

### 2.4. `monitor_performance` Decorator

A decorator that can be applied to any function to monitor its execution time.

**Functionality:**
-   Logs a `WARNING` if the function takes longer than 1.0 second to execute.
-   Logs a `DEBUG` message with the completion time for faster operations.
-   Logs an `ERROR` if the function fails, including the elapsed time before the failure.

### 2.5. `BatchProcessor` Class

A class for processing a list of items in parallel batches using a thread pool.

**Methods:**
-   **`__init__(self, batch_size: int = 100, max_workers: int = 4)`**: Initializes the batch processor with a specified batch size and number of worker threads.
-   **`process_items(self, items: list, processor: Callable, error_handler: Callable | None = None) -> list`**: Processes a list of items in parallel. It takes a `processor` function to apply to each item and an optional `error_handler` to manage exceptions.
-   **`process_items_async(self, items: list, processor: Callable) -> list`**: An asynchronous version of `process_items`.

### 2.6. `TextPreprocessor` Class

A class for centralized text preprocessing, designed to clean text once before chunking and indexing to improve performance.

**Key Features:**
-   **Caching**: Caches the results of text preparation to avoid reprocessing identical inputs.
-   **Type-Specific Cleaning**: Applies different cleaning rules based on the `text_type` (`email`, `attachment`, `document`).
-   **Metadata Generation**: Returns a tuple of the cleaned text and a dictionary of preprocessing metadata (e.g., original length, cleaned length, reduction ratio).

### 2.7. `should_skip_retrieval_cleaning()`

A utility function to determine if a document or chunk has already been pre-cleaned during the indexing process and should therefore skip the cleaning step during retrieval.

---

## 3. Dependencies

-   **`email_processing.clean_email_text`**: Used by the `TextPreprocessor` for cleaning email text.