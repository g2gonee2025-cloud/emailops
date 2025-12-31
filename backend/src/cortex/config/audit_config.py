#!/usr/bin/env python3
"""
Audit Configuration Script

Compares configuration defaults from Python models against environment variables.
This utility helps ensure that environment-specific overrides are intentional
and that there are no stale or unexpected configurations.

Key Features:
- Introspection-based: Directly imports and inspects Pydantic models, avoiding
  brittle regex parsing. This is architecturally robust.
- Security Hardened: Automatically redacts sensitive values (keys, secrets, URLs)
  to prevent accidental PII or secret leakage in logs.
- Dynamic & Portable: Automatically finds the project source root, so it can be
  run from any directory without modification.
- Graceful Error Handling: Catches common errors like missing files.
"""
import inspect
import os
import sys
from pathlib import Path
from typing import Any

# -----------------------------------------------------------------------------
# Path Setup: Add project root to sys.path for robust module imports
# -----------------------------------------------------------------------------


def get_project_root() -> Path:
    """
    Dynamically find the true project root directory.

    The key challenge is nested `pyproject.toml` files. The true root is the
    top-level directory that contains both `backend` and `frontend` folders.
    This is a stable architectural landmark.
    """
    try:
        # Start from the script's location and search upwards.
        current_path = Path(__file__).resolve().parent
        while (
            not (current_path / "backend").is_dir()
            or not (current_path / "frontend").is_dir()
        ):
            if current_path.parent == current_path:
                # Reached the filesystem root without finding the project structure.
                raise FileNotFoundError(
                    "Could not find the project root directory containing both 'backend' and 'frontend'."
                )
            current_path = current_path.parent
        return current_path
    except NameError:
        # Fallback for environments where __file__ is not defined
        cwd = Path.cwd()
        if (cwd / "backend").is_dir() and (cwd / "frontend").is_dir():
            return cwd
    raise FileNotFoundError(
        "Could not determine project root. Ensure script is run from within the project."
    )


def setup_sys_path() -> Path:
    """Find the project root and add the 'cortex' package source to sys.path."""
    project_root = get_project_root()
    # The 'cortex' package is located at 'backend/src/cortex'.
    # To import 'cortex.*', we must add its parent directory 'backend/src' to sys.path.
    src_path = project_root / "backend" / "src"
    cortex_init = src_path / "cortex" / "__init__.py"
    if not src_path.is_dir() or not cortex_init.is_file():
        raise FileNotFoundError(
            f"Source directory not found at expected path: {src_path}"
        )
    # Use str() for compatibility across Python versions.
    src_str = str(src_path)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)
    return project_root


from pydantic import BaseModel

# -----------------------------------------------------------------------------
# Core Logic
# -----------------------------------------------------------------------------


def parse_env_file(path: Path) -> dict[str, str]:
    """
    Parse a .env file into a dictionary. Handles comments and empty lines.
    """
    env_vars = {}
    if not path.is_file():
        print(
            f"Warning: .env file not found at '{path}'. Continuing without it.",
            file=sys.stderr,
        )
        return env_vars
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, val = line.split("=", 1)
                    env_vars[key.strip()] = val.strip()
    except OSError as exc:
        print(
            f"Warning: Unable to read .env file at '{path}': {exc}",
            file=sys.stderr,
        )
        return env_vars
    return env_vars


def parse_models_via_introspection(models_module: Any) -> dict[str, dict[str, Any]]:
    """
    Parse config models using introspection instead of brittle regex.

    This function mocks the `_env` helper to capture the default values
    defined in the Pydantic models, then triggers model instantiation to
    collect the data. This is a robust, architectural best practice.
    """
    definitions: dict[str, dict[str, Any]] = {}

    def mock_env(key: str, default: Any, value_type: type = str) -> Any:
        definitions[key] = {"default": default, "type": value_type.__name__}
        return default

    original_env = getattr(models_module, "_env", None)
    original_env_list = getattr(models_module, "_env_list", None)
    if not callable(original_env) or not callable(original_env_list):
        print(
            "Warning: models._env or models._env_list not available; "
            "introspection may be incomplete.",
            file=sys.stderr,
        )
        return definitions

    def mock_env_list(key: str, default: Any = "") -> list[str]:
        definitions[key] = {"default": default, "type": "list"}
        if default is None:
            return []
        if isinstance(default, list):
            return default
        if isinstance(default, tuple):
            return list(default)
        return [part.strip() for part in str(default).split(",") if part.strip()]

    models_module._env = mock_env
    models_module._env_list = mock_env_list

    try:
        config_classes: list[type[BaseModel]] = [
            obj
            for obj in vars(models_module).values()
            if inspect.isclass(obj)
            and issubclass(obj, BaseModel)
            and obj is not BaseModel
        ]
        for model_class in config_classes:
            try:
                model_class.model_validate({})
            except Exception as exc:
                print(
                    f"Warning: Failed to introspect {model_class.__name__}: {exc}",
                    file=sys.stderr,
                )
                continue
    finally:
        models_module._env = original_env
        models_module._env_list = original_env_list
    return definitions


def _normalize_value(val: Any) -> str:
    """Normalize a value for comparison (strip whitespace and quotes)."""
    if val is None:
        return ""
    if isinstance(val, bool):
        return "true" if val else "false"
    if isinstance(val, (int, float)):
        return str(val)
    text = str(val).strip()
    if (text.startswith('"') and text.endswith('"')) or (
        text.startswith("'") and text.endswith("'")
    ):
        text = text[1:-1].strip()
    return text


def _determine_status(
    model_def: dict | None, env_val: str | None, code_default: Any
) -> str:
    """Determine the status of a config key comparison."""
    if model_def and env_val is not None:
        if _normalize_value(code_default) == _normalize_value(env_val):
            return "MATCH"
        return "MISMATCH"
    if model_def and env_val is None:
        return "CODE_ONLY"
    if not model_def:
        return "ENV_ONLY"
    return "UNKNOWN"


def _should_print_entry(status: str, key: str) -> bool:
    """Determine if an entry should be printed based on its status."""
    if status == "MISMATCH":
        return True
    if status == "ENV_ONLY" and key.isupper():
        return True
    return False


_PREFIXES = ("OUTLOOKCORTEX_", "EMAILOPS_")


def _strip_prefix(key: str) -> str:
    for prefix in _PREFIXES:
        if key.startswith(prefix):
            return key[len(prefix) :]
    return key


def _get_env_value(env_vars: dict[str, str], key: str) -> str | None:
    """Get environment value, checking for standard prefixes first."""
    if key.startswith(_PREFIXES):
        return env_vars.get(key)
    prefixed_key_new = f"OUTLOOKCORTEX_{key}"
    if prefixed_key_new in env_vars:
        return env_vars[prefixed_key_new]
    prefixed_key_legacy = f"EMAILOPS_{key}"
    if prefixed_key_legacy in env_vars:
        return env_vars[prefixed_key_legacy]
    return env_vars.get(key)


def _is_sensitive(key: str) -> bool:
    """Check if a key is likely to be sensitive."""
    key_lower = key.lower()
    sensitive_patterns = [
        "auth",
        "jwt",
        "key",
        "secret",
        "password",
        "token",
        "url",
        "credential",
        "private",
        "cert",
        "api",
    ]
    return any(pattern in key_lower for pattern in sensitive_patterns)


def main() -> int:
    """Main execution function."""
    try:
        project_root = setup_sys_path()
    except Exception as exc:
        print(f"Error: Could not set python path: {exc}", file=sys.stderr)
        print(
            "Please ensure the script is run from within the project structure.",
            file=sys.stderr,
        )
        return 1

    try:
        from cortex.config import models as config_models
    except ImportError as exc:
        print(f"Error: Could not import cortex.config.models: {exc}", file=sys.stderr)
        return 1

    env_file_vars = parse_env_file(project_root / ".env")
    env_vars = {**env_file_vars, **os.environ}
    model_defs = parse_models_via_introspection(config_models)

    print(f"{'KEY':<35} | {'CODE DEFAULT':<40} | {'ENV VALUE':<40} | {'STATUS':<15}")
    print("-" * 135)

    env_base_keys = {_strip_prefix(k) for k in env_vars if k.startswith(_PREFIXES)}
    env_base_keys |= {k for k in env_vars if k in model_defs}

    app_keys = set(model_defs.keys()) | env_base_keys

    for key in sorted(app_keys):
        model_def = model_defs.get(key)
        env_val = _get_env_value(env_vars, key)
        code_default_val = model_def.get("default") if model_def else None
        code_default = str(code_default_val) if model_def else "N/A"
        env_display = str(env_val) if env_val is not None else "N/A"

        status = _determine_status(model_def, env_val, code_default_val)

        if _is_sensitive(key):
            if model_def and code_default_val not in (None, ""):
                code_default = "***REDACTED***"
            if env_val is not None and env_val != "":
                env_display = "***REDACTED***"

        if _should_print_entry(status, key):
            print(f"{key:<35} | {code_default:<40} | {env_display:<40} | {status:<15}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
