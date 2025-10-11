# `llm_client.py` - Dynamic LLM Runtime Shim

## 1. Overview

Like `env_utils.py`, this file is a **compatibility shim**. However, it is more advanced and dynamic. Its purpose is to provide a stable, backward-compatible API for interacting with Large Language Models (LLMs), while transparently forwarding all requests to the modern, unified `llm_runtime.py` module.

It ensures that legacy code continues to work without modification, while also allowing new features from the runtime to be immediately accessible.

## 2. Core Mechanism: Dynamic Forwarding

This module uses Python's powerful `__getattr__` feature to dynamically handle attribute access. When you try to use a function from `llm_client`, it catches the request and forwards it to `llm_runtime`.

This means you can call any function from `llm_runtime` through `llm_client`, even if it's not explicitly defined here.

```mermaid
graph TD
    subgraph "Your Code"
        A[Call llm_client.some_function()]
    end

    subgraph "llm_client.py"
        B{Is some_function defined here?};
        B -- No --> C[__getattr__('some_function') is triggered];
        C --> D[Looks for 'some_function' in llm_runtime.py];
    end

    subgraph "llm_runtime.py"
        E[Provides the actual some_function]
    end

    A --> B;
    D --> E;
```

## 3. Provided Functions

The shim explicitly defines a few key functions for clarity and backward compatibility.

### 3.1. Main API

These are thin wrappers that pass all arguments directly to the runtime.

-   `complete_text(...)`: For text generation.
-   `complete_json(...)`: For generating structured JSON output.
-   `embed_texts(...)`: For creating vector embeddings from text.

### 3.2. Compatibility Aliases

To support older code, the following aliases are provided:

-   `complete(...)` -> `complete_text(...)`
-   `json_complete(...)` -> `complete_json(...)`
-   `embed(...)` -> `embed_texts(...)`

### 3.3. The `embed()` Safety Net

The `embed()` alias includes an important safety feature. It's a common mistake to pass a single string to an embedding function that expects a list of strings. This shim catches that error and provides a helpful message.

```mermaid
graph TD
    A[Call embed("a single string")] --> B{Is input a string or bytes?};
    B -- Yes --> C[Raise TypeError with helpful message: "wrap in a list, e.g., embed([text])"];
    B -- No --> D[Proceed to call embed_texts(...)];
```

## 4. Developer Guidance

This file is designed for API compatibility, not for implementing logic.

**==> For the actual implementation of any LLM function, refer to `emailops/llm_runtime.py`.**