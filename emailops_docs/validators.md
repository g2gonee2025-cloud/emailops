# `validators.py` — Path & Command Validation Utilities

> **Purpose:** Centralized validation/sanitization helpers for paths, commands, and basic identifiers used throughout EmailOps. While security testing is out of scope for this exercise, the helpers exist and are documented here for completeness.

---

## 1) Paths

### `validate_directory_path(path, must_exist=True, allow_parent_traversal=False) -> (bool, str)`
- Blocks `..` traversal (unless allowed) by inspecting `Path.parts`.
- Expands `~` and resolves to an absolute canonical path.
- Optionally asserts existence and directory type.

### `validate_file_path(path, must_exist=True, allowed_extensions=None, allow_parent_traversal=False) -> (bool, str)`
- Same traversal protections as directories.
- Optional extension allow‑list (case‑insensitive).
- Asserts file type on existence.

### `sanitize_path_input(path_input: str) -> str`
- Removes null bytes, trims, and strips shell metacharacters. Keeps `[a-zA-Z0-9._\-/\\: ]`.

---

## 2) Command Arguments

### `validate_command_args(command: str, args: list[str], allowed_commands: list[str] | None) -> (bool, str)`
- Optional whitelist.
- Rejects `; | & $ \\` backticks and newlines in command or args.
- Detects null bytes.

### `quote_shell_arg(arg: str) -> str`
- POSIX‑safe quoting via `shlex.quote`.

---

## 3) Identifiers

- `validate_project_id(project_id: str)` — checks length, pattern, and hyphen rules for GCP projects.  
- `validate_environment_variable(name: str, value: str)` — checks uppercase name pattern and null‑byte free value.  
- `validate_email_format(email: str)` — basic RFC‑style validation with length caps.

