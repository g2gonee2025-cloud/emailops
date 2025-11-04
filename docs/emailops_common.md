# `emailops.common`

**Primary Goal:** To provide a set of shared, foundational data structures and types that are used consistently across the entire EmailOps application. This package is critical for ensuring data integrity, type safety, and robust error handling.

## Directory Mapping

```
.
└── emailops/
    └── common/
        ├── __init__.py
        ├── models.py
        └── types.py
```

---

## Core Components & Connections

This package is composed of two main files: `models.py` for data structures and `types.py` for a type-safe error handling mechanism.

### `common.models`: The `Participant` Model

- **Purpose:** To establish a single, canonical, and validated representation of an email participant. The docstring explicitly states that this `Pydantic` model replaces several inconsistent legacy formats (tuples, dicts), which is a common source of bugs in data processing pipelines.
- **Key Features:**
    - **Validation:** It uses `Pydantic` for automatic, runtime validation. For example, the `email` field is validated using `EmailStr` to ensure it's a correctly formatted email address, and the `name` field is validated to ensure it's not empty. This prevents invalid data from propagating through the system.
    - **Normalization:** The field validators also perform normalization, such as converting emails to lowercase, which is crucial for reliable deduplication.
    - **Backward Compatibility:** The `from_tuple` and `from_dict` class methods provide a clean way to convert data from the old, inconsistent formats into the new, validated `Participant` model. This is a key feature for migrating a legacy codebase without breaking everything at once.
    - **Schema Definition:** The model acts as a clear, self-documenting schema for what constitutes a "participant," including their name, email, role, and optional fields like tone and title.
- **Connections:** This model is used extensively in modules that deal with conversation metadata, such as `core_manifest.extract_participants_detailed`, ensuring that any part of the system that needs to know "who" was in a conversation gets a consistent and reliable data structure.

### `common.types`: The `Result[T, E]` Type

- **Purpose:** To provide a generic, type-safe container for the result of an operation that might fail. This pattern (also known as `Either` or `Maybe` in other languages) is a powerful alternative to returning `None`, raising exceptions for non-exceptional errors, or using ambiguous tuples like `(bool, str)`.
- **Key Features:**
    - **Explicit Success/Failure:** An instance of `Result` is either a `Success` (containing a `value` of type `T`) or a `Failure` (containing an `error` of type `E`). This forces the developer who calls the function to explicitly handle both possibilities, making it much harder to write code that ignores errors.
    - **Type Safety:** When used with a type checker like MyPy, the `Result` type allows for compile-time verification. If you check `if result.ok:`, the type checker knows that inside that block, `result.value` is guaranteed to be of type `T` and not `None`.
    - **Composability:** The class provides several methods for chaining and transforming results in a clean, functional style:
        - **`.map(fn)`:** Applies a function to the `value` if it's a `Success`, otherwise it passes the `error` through.
        - **`.and_then(fn)`:** Chains another function that also returns a `Result`, allowing for multi-step operations where any step can fail.
        - **`.unwrap_or(default)`:** Provides a safe way to get the value or a default if the operation failed.
- **Connections:** This `Result` type is used by the validation functions in `core_validators` (e.g., `validate_directory_result`). This is a perfect use case, as path validation can fail for many predictable reasons (doesn't exist, not a directory, etc.), and `Result` provides a clean way to return either the validated `Path` object or a descriptive error string.

---

## Key Design Patterns

- **Data Modeling with Pydantic:** The use of `Pydantic` for the `Participant` model is a modern best practice in Python. It provides the benefits of dataclasses (concise definition) with the added power of runtime validation and serialization, which is crucial for building reliable data-intensive applications.
- **Railway Oriented Programming (The `Result` type):** The `Result` object is the core of this pattern. It encourages a style of programming where functions are chained together, and errors are handled explicitly at the boundaries. The "two tracks" of the railway are the success track and the failure track. An operation stays on the success track as long as everything works, but as soon as one function returns a `Failure`, the rest of the chain is bypassed, and the error is passed straight through to the end. This leads to cleaner, more robust, and easier-to-reason-about code.
- **Centralized Types:** By defining these core data structures in a `common` package, the project avoids circular dependencies and establishes a "common language" for the rest of the application modules to use when passing data or signaling errors.