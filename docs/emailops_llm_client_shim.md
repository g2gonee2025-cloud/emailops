# `emailops.llm_client_shim`

**Primary Goal:** To act as a stable, backward-compatible abstraction layer (a "shim") over the actual language model implementation. This module decouples the rest of the application from the specific details of `llm_runtime.py`, allowing the underlying runtime to be updated or changed without breaking the code that calls it.

## Directory Mapping

```
.
└── emailops/
    ├── llm_client_shim.py
    └── llm_runtime.py
```

---

## Core Functions & Connections

This module provides a simple, consistent API for the most common LLM operations. Instead of containing the logic themselves, these functions are thin wrappers that dynamically forward calls to the actual implementation in `llm_runtime.py`.

### `complete_text(...)`
- **Purpose:** Provides a standard way to get a plain text completion from a language model.
- **Connection:** It dynamically looks up and calls `llm_runtime.complete_text()` every time it is invoked.

### `complete_json(...)`
- **Purpose:** Provides a way to get a structured JSON completion from a language model, often by passing a schema that the model is instructed to follow.
- **Connection:** It dynamically looks up and calls `llm_runtime.complete_json()`.

### `embed_texts(...)`
- **Purpose:** Provides a standard way to convert a batch of texts into numerical vector embeddings.
- **Connection:** It dynamically looks up and calls `llm_runtime.embed_texts()`.

---

## Dynamic Attribute Forwarding

The power of this shim comes from its use of Python's module-level `__getattr__` and `__dir__` (defined in PEP 562). This is a sophisticated feature that makes the module act like a dynamic proxy.

- **`__getattr__(name)`:** If another module tries to import a name that isn't explicitly defined in this file (like `LLMError` or a newly added function), this function intercepts the request. It then tries to find that attribute on the `_rt` (the `llm_runtime` module) and returns it. This means any function or class added to `llm_runtime` automatically becomes available through `llm_client_shim` without any changes needed here.

- **`__dir__()` and `__getattr__("__all__")`:** These functions work together to provide proper support for tools like IDEs (for autocompletion) and `from emailops.llm_client_shim import *`. They dynamically build a list of all public names available from both the shim itself and the underlying runtime, ensuring a clean and accurate public API surface.

---

## Key Design Patterns

- **Shim / Facade Pattern:** This module is a classic example of a **Shim**. It provides a simplified, stable API that hides the more complex or changing implementation details of the underlying `llm_runtime`. It acts as a facade, presenting a clean interface to the rest of the application.

- **Dynamic Proxy:** The use of `__getattr__` effectively turns this module into a dynamic proxy for `llm_runtime`. It intercepts calls, forwards them to the real implementation, and makes the separation between the interface and the implementation transparent to the caller.

- **Decoupling:** The primary benefit of this design is **decoupling**. The core application logic (in `feature_search_draft`, `feature_summarize`, etc.) depends only on the stable `llm_client_shim` API. The `llm_runtime` can be refactored, have its dependencies changed, or even be swapped out entirely, and as long as it continues to provide the functions expected by the shim, the rest of the application will continue to work without modification. This is a powerful technique for managing complexity and reducing maintenance overhead in large systems.

- **Backward Compatibility:** As stated in the docstring, this module is explicitly designed to provide a stable API for any legacy code that might have depended on it, even as the underlying runtime evolves. The dynamic forwarding ensures that both old and new functions remain accessible through this single, consistent point of entry.