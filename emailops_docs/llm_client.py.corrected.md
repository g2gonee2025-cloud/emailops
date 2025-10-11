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

## 3. Internal Helper Functions

The module includes several internal helper functions that manage the dynamic forwarding:

### 3.1. `_rt_attr(name: str) -> Any`

A helper function that resolves attributes on the runtime module with consistent error messaging. If the attribute doesn't exist in `llm_runtime`, it raises an `AttributeError` with a clear message.

### 3.2. `_runtime_exports() -> list[str]`

Fetches the runtime's declared public exports dynamically (no caching). This function:
- Retrieves `__all__` from `llm_runtime` if it exists
- Validates that entries are strings
- Handles various iterable types (list, tuple, or any Iterable)
- Returns an empty list if runtime doesn't declare a usable `__all__`

## 4. Provided Functions

The shim explicitly defines a few key functions for clarity and backward compatibility.

### 4.1. Main API

These are thin wrappers that pass all arguments directly to the runtime using `_rt_attr()` for resolution:

-   **`complete_text(...)`**: For text generation.
-   **`complete_json(...)`**: For generating structured JSON output.
-   **`embed_texts(...)`**: For creating vector embeddings from text.

### 4.2. Compatibility Aliases

To support older code, the following aliases are provided:

-   **`complete(...)`** -> `complete_text(...)`
-   **`json_complete(...)`** -> `complete_json(...)`
-   **`embed(...)`** -> `embed_texts(...)`

### 4.3. The `embed()` Safety Net

The `embed()` alias includes an important safety feature. It's a common mistake to pass a single string to an embedding function that expects a list of strings. This shim catches that error and provides a helpful message.

```mermaid
graph TD
    A[Call embed("a single string")] --> B{Is input a string, bytes, bytearray, or memoryview?};
    B -- Yes --> C[Raise TypeError with helpful message: "wrap in a list, e.g., embed([text])"];
    B -- No --> D[Convert to list if not already];
    D --> E{Are all elements strings?};
    E -- No --> F[Raise TypeError: "expects an iterable of str"];
    E -- Yes --> G[Proceed to call embed_texts(...)];
```

## 5. Dynamic Export Management

The module implements sophisticated export management to maintain a clean and consistent API surface.

### 5.1. Core Exports List

The `_CORE_EXPORTS` list defines the shim's guaranteed exports:
- Core functions: `complete_text`, `complete_json`, `embed_texts`
- Compatibility aliases: `complete`, `json_complete`, `embed`
- Error class: `LLMError` (conditionally included based on runtime availability)

### 5.2. Dynamic `__all__` Construction

The `__getattr__` function handles requests for `__all__` specially:
1. Builds core exports list, excluding `LLMError` if not present in runtime
2. Merges with runtime's declared exports from `_runtime_exports()`
3. Deduplicates while preserving order
4. Returns the merged list for clean `import *` behavior

### 5.3. `__dir__()` Implementation

Provides clean IDE/tooling completion by returning a sorted, deduplicated list of all available attributes. This ensures autocomplete and introspection tools work correctly.

## 6. Type Checking Support

The module includes a `TYPE_CHECKING` block that:
- Only executes during static type analysis (not at runtime)
- Imports type hints from `llm_runtime` for better IDE support
- Has no runtime cost but helps type checkers infer correct signatures

```python
if TYPE_CHECKING:
    # Import for type checkers only (no runtime cost)
    from .llm_runtime import (  # noqa: F401
        complete_text as _complete_text_t,
    )
```

## 7. Error Handling

The module provides clear error messages through:
- `_rt_attr()`: Consistent error messages for missing runtime attributes
- Custom `AttributeError` in `__getattr__`: Clear module-level attribute errors
- Type validation in `embed()`: Helpful messages for common usage mistakes

## 8. Developer Guidance

This file is designed for API compatibility, not for implementing logic.

**Important Notes:**
- All actual LLM logic is in `emailops/llm_runtime.py`
- This shim ensures backward compatibility while allowing runtime evolution
- The dynamic forwarding means new runtime features are immediately available
- Type checkers get proper hints despite the dynamic nature

**For implementation details, refer to `emailops/llm_runtime.py`.**