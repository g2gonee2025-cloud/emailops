# EmailOps: System Architecture and Orchestration

## 1. Overview

The EmailOps application is a modular system designed for resilience and high-quality AI-powered email processing. The architecture separates the main application entry points (orchestrators) from the core AI engine and low-level utilities. This design allows for clear separation of concerns and makes the system easier to maintain and extend.

This document provides a high-level overview of how these components work together.

---

## 2. High-Level Architecture Diagram

This diagram illustrates the primary components and the flow of data and control between them.

```mermaid
graph TD
    subgraph "User Interaction"
        A[User via CLI]
    end

    subgraph "Orchestration Layer"
        B[search_and_draft.py];
        C[summarize_email_thread.py];
        D[doctor.py];
    end

    subgraph "Core Logic Layer"
        E[llm_runtime.py];
        F[validators.py];
        G[config.py];
    end

    subgraph "Data & Utility Layer"
        H[utils.py];
        I[qdrant_client.py];
        J[Vector Index <br/>(FAISS or Qdrant)];
        K[Email Data <br/>(Filesystem)];
    end

    A --> B;
    A --> C;
    A --> D;

    B --> E;
    B --> H;
    B --> F;
    C --> E;
    C --> H;
    D --> E;

    E --> G;
    E --> J;
    H --> K;
    I --> J;
```

---

## 3. Component Roles

-   **Orchestration Layer** (`search_and_draft.py`, `summarize_email_thread.py`):
    -   These are the main user-facing scripts that define the high-level workflows (e.g., "draft a reply," "summarize a thread"). They parse command-line arguments and orchestrate calls to the other layers to accomplish a specific task.

-   **Core Logic Layer** (`llm_runtime.py`, `validators.py`, `config.py`):
    -   `llm_runtime.py`: The absolute heart of the application. It handles all interactions with LLM and embedding providers, featuring a sophisticated resilience engine for project rotation and retries. All AI capabilities flow through this module.
    -   `validators.py`: A critical security component that sanitizes and validates all external inputs, such as file paths and command arguments, to prevent common vulnerabilities.
    -   `config.py`: Provides a centralized, singleton pattern for managing all application configuration, loading from environment variables with sensible defaults.

-   **Data & Utility Layer** (`utils.py`, `qdrant_client.py`, Data Stores):
    -   `utils.py`: Provides robust, low-level functions for reading files and extracting text from various formats (PDFs, Word docs, etc.). It is the primary interface to the raw email data on the filesystem.
    -   `qdrant_client.py`: An optional client for interacting with a Qdrant vector database, used for storing and searching embeddings at scale. It also includes a utility for migrating from a local FAISS index.
    -   **Vector Index & Email Data**: These represent the data stores themselves—the raw email files on the filesystem and the vector index (either local FAISS files or a Qdrant database) used for similarity search.

-   **Compatibility Shims** (`llm_client.py`, `env_utils.py`):
    -   These files are not primary components but serve as backward-compatibility layers. They simply import and re-export logic from the modern `llm_runtime.py`, ensuring that older parts of the codebase continue to function without modification.

---

## 4. Example End-to-End Workflow: Drafting a Reply

This workflow illustrates how the components work together to execute a common task.

1.  **Initiation**: A user runs `python -m emailops.search_and_draft --reply-conv-id "C123XYZ"`.

2.  **Validation**: `search_and_draft.py` uses `validators.py` to ensure the provided root path is safe.

3.  **Load Conversation**: It calls `_load_conv_data`, which in turn uses `utils.py` to read `Conversation.txt` and the associated `manifest.json` from the specified directory.

4.  **Derive Query**: A search query is automatically generated from the content of the last inbound message in the conversation.

5.  **Gather Context (RAG)**: The `_gather_context_for_conv` function is called.
    -   It sends the query to `llm_runtime.py` to be converted into a vector embedding.
    -   It searches the vector index (e.g., FAISS) for similar document chunks within the same `conv_id`.
    -   It applies a recency boost to the search scores and collects the most relevant text snippets until a token budget is met.

6.  **Generate Draft**: The query and the retrieved context snippets are passed to the `draft_email_structured` function.
    -   This triggers the **"Draft, Critique, Audit"** multi-agent workflow, which involves several sequential calls to `llm_runtime.py` to generate, review, and refine the email draft.

7.  **Construct Email**: The final, audited draft is passed to the `_build_eml` function.
    -   It determines the correct `To`, `CC`, and `Subject` headers from the original conversation data.
    -   It calls `select_relevant_attachments` to choose which files to attach based on the context.
    -   It uses Python's `email` library to construct a complete, multipart `.eml` message with headers, body, and attachments.

8.  **Save Output**: The generated `.eml` bytes are saved to the filesystem.
