# `file_utils.py` - File System Utilities
## 1. Overview

This module provides a set of robust and efficient utilities for common file system operations. It includes functions for reading text files with encoding detection, ensuring directory existence, finding specific project directories, and handling file locks in a platform-aware manner.

**Key Features:**
- **Robust File Reading**: Safely reads text files with multiple encoding fallbacks.
- **Performance**: Uses LRU caching for file encoding detection.
- **Concurrency Safe**: Provides a context manager for file locking.
- **Directory Management**: Includes helpers for creating and finding specific directories.
- **Temporary Resources**: A context manager for creating and cleaning up temporary directories.

---

## 2. Core Functions

### 2.1. Text and File Reading

#### `read_text_file(path: Path, *, max_chars: int | None = None) -> str`
Reads a text file with robust encoding detection and content sanitization.

**Workflow:**
1.  **Encoding Detection**: Calls `_get_file_encoding()` to determine the file's encoding.
2.  **File Read**: Opens the file with the detected encoding and `errors="ignore"`.
3.  **Truncation**: If `max_chars` is provided, it reads only up to that many characters.
4.  **Sanitization**: Passes the content through `_strip_control_chars()` to remove non-printable characters and normalize newlines.
5.  **Error Handling**: Returns an empty string if any exception occurs during the process.

#### `_get_file_encoding(path: Path) -> str`
Detects the most likely encoding of a file by attempting to read it with a list of common encodings.

**Features:**
-   **LRU Cache**: Caches the result to avoid re-detecting the encoding for the same file. The cache size is configurable via the `FILE_ENCODING_CACHE_SIZE` environment variable (default: 1024).
-   **Encoding Priority**: Tries encodings in the following order: `utf-8-sig`, `utf-8`, `utf-16`, `latin-1`.
-   **Fallback**: Returns `latin-1` as a safe fallback that is unlikely to raise a `UnicodeDecodeError`.

#### `_strip_control_chars(s: str) -> str`
A utility function to remove non-printable ASCII control characters from a string and normalize line endings (CRLF/CR to LF).

### 2.2. Directory Operations

#### `ensure_dir(p: Path) -> None`
Ensures that a directory exists, creating it and any necessary parent directories if it does not. This is an idempotent operation.

#### `find_conversation_dirs(root: Path) -> list[Path]`
A specialized function that recursively searches a root directory for "conversation directories." A directory is considered a conversation directory if it contains a file named `Conversation.txt`. It returns a sorted list of the parent directories.

### 2.3. Resource Management

#### `temporary_directory(prefix: str = "emailops_")`
A context manager that creates a temporary directory and ensures it is automatically cleaned up (removed) upon exiting the context, even if errors occur.

**Example:**
```python
from emailops.file_utils import temporary_directory

with temporary_directory() as temp_dir:
    # Use the temporary directory
    print(f"Created temporary directory: {temp_dir}")
    # ... write files, perform operations ...

# The directory is automatically removed here
```

#### `file_lock(path: Path, timeout: float = 10.0)`
A context manager for creating a file-based lock to prevent race conditions when multiple processes might access the same file.

**Features:**
-   **Platform-Aware**: Uses `msvcrt` for file locking on Windows and `fcntl` on Unix-like systems.
-   **Timeout**: Raises a `TimeoutError` if the lock cannot be acquired within the specified timeout.
-   **Automatic Cleanup**: The lock file (`.lock`) is automatically created on entering the context and removed on exiting.

**Example:**
```python
from emailops.file_utils import file_lock

file_to_protect = Path("my_shared_file.dat")

try:
    with file_lock(file_to_protect, timeout=5.0):
        # Safely perform operations on the file
        content = file_to_protect.read_text()
        # ... modify and write ...
except TimeoutError:
    print("Could not acquire lock on the file.")
```

---

## 3. Dependencies

-   **`msvcrt`**: (Windows only) for file locking.
-   **`fcntl`**: (Unix-like only) for file locking.

The module is designed to work on both Windows and Unix-like systems by checking `sys.platform`.