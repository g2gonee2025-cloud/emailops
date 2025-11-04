# `emailops.core_validators`

**Primary Goal:** To provide a centralized set of security-focused validation and sanitization functions. This module is a critical defense layer, responsible for ensuring that all external inputs—such as file paths, command-line arguments, and environment variables—are safe before they are used by the application.

## Directory Mapping

```
.
└── emailops/
    └── core_validators.py
```

---

## Core Functions & Connections

This module is a toolbox of functions designed to be called by other parts of the system whenever they accept external input.

### Path Validation

- **`validate_directory_path(path, ...)` & `validate_file_path(path, ...)`:**
    - **Purpose:** These are the workhorse functions for validating file system paths. They perform a series of critical security checks.
    - **Security Checks:**
        1.  **Parent Traversal (`../`):** By default, they reject any path containing `..` segments. This is a crucial defense against path traversal attacks, where an attacker might try to access files outside of the intended directory (e.g., `/etc/passwd`).
        2.  **Existence & Type:** They check if the path exists and is of the correct type (a directory or a file).
        3.  **Absolute Path Requirement:** They ensure the path resolves to an absolute path, preventing ambiguity.
        4.  **Symbolic Link (Symlink) Attacks:** They check if the path is a symlink that points to an unexpected location.
    - **TOCTOU Warning:** The docstrings correctly note the "Time-of-Check to Time-of-Use" (TOCTOU) race condition. This means that even if a path is valid at the moment of the check, an attacker could potentially change it (e.g., replace a file with a symlink) before the application actually uses it. The proper mitigation, as noted, is for the calling code to also handle exceptions during the actual file operation.
- **Connections:** These functions are used by `emailops.cli` to validate the `--root` argument and by any other component that accepts a file or directory path from an external source.

### Command & Argument Validation

- **`validate_command_args(command, args, ...)`:**
    - **Purpose:** To prevent shell injection vulnerabilities when the application needs to run external subprocesses.
    - **Security Checks:**
        1.  **Sanitization:** It first uses `sanitize_path_input` to strip potentially dangerous characters from the command and its arguments.
        2.  **Blocklist:** It checks the command against a blocklist of inherently dangerous commands (`rm`, `mv`, `shutdown`, etc.).
        3.  **Whitelist (Optional):** It can also validate the command against a specific `allowed_commands` list for stricter control.
        4.  **Shell Injection Patterns:** It scans both the command and its arguments for characters that have special meaning in a shell, such as `;`, `|`, `&`, and `$`, which could be used to chain malicious commands.
- **Connections:** This function is a critical security gate called by `emailops.cli._run_email_indexer` before it constructs and executes the `subprocess.run` call.

### Input Sanitization & Formatting

- **`sanitize_path_input(path_input)`:** A simple but important function that removes a predefined set of dangerous characters from a string, making it safer to use as a path.
- **`quote_shell_arg(arg)`:** A wrapper around Python's `shlex.quote`, which is the standard, correct way to make a string safe for inclusion in a shell command line.

### Format-Specific Validators

- **`validate_email_format(email)`:** Uses a regular expression to perform a reasonably strict check on the format of an email address.
- **`validate_project_id(project_id)`:** Enforces the specific formatting rules for Google Cloud project IDs (length, character set, starting character, etc.).
- **`validate_environment_variable(name, value)`:** Checks that environment variable names and values are well-formed and do not contain dangerous characters like null bytes.

---

## Evolving API Design: Tuples -> `ValidationResult` -> `Result[T, E]`

This module showcases a thoughtful evolution of its API design, aimed at improving type safety and ergonomics.

1.  **Original API (preserved):** Functions like `validate_directory_path` return a `tuple[bool, str]`. This is common in Python but has a weakness: the caller can accidentally ignore the boolean `is_valid` flag and use the message string incorrectly.
2.  **Ergonomic Variant:** Functions like `validate_directory_path_info` were introduced. They return a `ValidationResult` dataclass. This is slightly better as it names the fields (`ok`, `msg`, `value`), but still relies on the caller to check the `ok` flag.
3.  **Type-Safe `Result` Pattern (current best practice):** The newest functions, like `validate_directory_result`, return a `Result[Path, str]`. This is a powerful pattern (common in languages like Rust) that forces the caller to handle both the success and failure cases, often at compile time with a type checker like MyPy. To get the `Path` value, the caller *must* first check if the result is `ok` or `unwrap` it, which makes it much harder to write code that ignores errors. This represents a significant improvement in the robustness and safety of the codebase.

---

## Key Design Patterns

- **Security by Default:** The validators are designed with a "deny by default" posture. For example, parent traversal in paths is disallowed unless explicitly enabled. This is a core principle of secure software design.
- **Pre-compiled Regex:** Like other core modules, this one pre-compiles all regular expression patterns at module load time for performance.
- **Centralized Validation Logic:** By concentrating all security validation logic in this one module, the application avoids having scattered, inconsistent, or incomplete security checks. It creates a single, auditable point of control for input validation.