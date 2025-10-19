# `processor.py` - EmailOps Orchestrator

## 1. Overview

This module serves as the main command-line interface (CLI) and orchestrator for the EmailOps application. It provides a unified entry point for various high-level commands, delegating the actual work to the specialized modules within the `emailops` package.

**Key Responsibilities:**
-   **Command-Line Parsing**: Defines and parses CLI arguments for all supported commands.
-   **Orchestration**: Calls the appropriate functions from other modules based on the user's command.
-   **Environment Setup**: Ensures that the necessary environment variables are configured for all operations.
-   **Error Handling**: Provides a centralized place for catching and logging exceptions.
-   **Parallel Processing**: Manages multiprocessing for tasks that can be parallelized, like summarizing multiple conversations.

---

## 2. Commands and Workflows

The processor module supports the following commands:

### 2.1. `index`
-   **Purpose**: Build or update the vector search index.
-   **Delegates to**: `email_indexer.py` (run as a separate subprocess).
-   **Workflow**:
    1.  Validates the `--root` directory.
    2.  Constructs the command-line arguments for `email_indexer.py`.
    3.  Runs the indexer in a subprocess with a timeout.
    4.  Checks the return code and logs any errors.

### 2.2. `reply`
-   **Purpose**: Draft a reply to a specific email conversation.
-   **Delegates to**: `search_and_draft.draft_email_reply_eml()`.
-   **Workflow**:
    1.  Validates the `--root` and `--conv-id` arguments.
    2.  Calls `draft_email_reply_eml()` with the provided parameters.
    3.  Saves the resulting `.eml` file to the specified output path or a default location.

### 2.3. `fresh`
-   **Purpose**: Draft a new email from scratch.
-   **Delegates to**: `search_and_draft.draft_fresh_email_eml()`.
-   **Workflow**:
    1.  Validates required arguments (`--to`, `--subject`, `--query`).
    2.  Calls `draft_fresh_email_eml()`.
    3.  Saves the resulting `.eml` file.

### 2.4. `chat`
-   **Purpose**: Engage in a conversational chat with the email data.
-   **Delegates to**: `search_and_draft.chat_with_context()`.
-   **Workflow**:
    1.  Performs a search using `_low_level_search` to gather context.
    2.  Calls `chat_with_context()` with the query and retrieved context.
    3.  Prints the response as JSON or plain text.

### 2.5. `summarize`
-   **Purpose**: Summarize a single email conversation.
-   **Delegates to**: `summarize_email_thread.analyze_conversation_dir()`.
-   **Workflow**:
    1.  Validates the conversation directory path.
    2.  Calls `analyze_conversation_dir()`.
    3.  Saves the analysis as `analysis.json` and optionally `analysis.md`.

### 2.6. `summarize-many`
-   **Purpose**: Summarize multiple conversations in parallel.
-   **Delegates to**: `_summarize_worker()` function, which calls `summarize_email_thread.analyze_conversation_dir()`.
-   **Workflow**:
    1.  Finds all conversation directories under the `--root`.
    2.  Creates a `_SummJob` for each conversation.
    3.  Uses a `multiprocessing.Pool` to execute `_summarize_worker` for each job in parallel.
    4.  Logs the success or failure of each job.

---

## 3. Key Implementation Details

### 3.1. Environment and Configuration
-   **`_ensure_env()`**: This function is called before executing commands to ensure that the environment is properly configured. It uses `get_config().update_environment()` to propagate all settings from the `EmailOpsConfig` object to `os.environ`.

### 3.2. Subprocess Execution
-   **`_run_email_indexer()`**: This function runs the `email_indexer.py` script as a separate Python process. This is done to isolate the `argparse` and `sys.argv` of the main processor from the indexer script.
-   **Security**: It includes command validation to prevent shell injection, either through the `validators` module or a fallback mechanism.

### 3.3. Parallel Processing
-   **`summarize-many`**: This command uses `multiprocessing.get_context("spawn")` to create a new process for each worker, which is a safe way to handle multiprocessing, especially on Windows.
-   **`_summarize_worker()`**: This top-level function is designed to be "picklable" so it can be sent to worker processes. It takes a `_SummJob` dataclass as its argument.

### 3.4. Custom Exceptions
The module defines its own set of custom exceptions to provide more specific error information:
-   `ProcessorError`
-   `IndexNotFoundError`
-   `ConfigurationError`
-   `CommandExecutionError`

---

## 4. Dependencies

This module acts as an orchestrator and has dependencies on most of the other core modules in the `emailops` package:
-   `config`
-   `index_metadata`
-   `search_and_draft`
-   `summarize_email_thread`
-   `validators` (optional, with fallback)