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
import sys
from pathlib import Path
from typing import Any, Optional

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
    try:
        project_root = get_project_root()
        # The 'cortex' package is located at 'backend/src/cortex'.
        # To import 'cortex.*', we must add its parent directory 'backend/src' to sys.path.
        src_path = project_root / "backend" / "src"
        if not src_path.is_dir():
            raise FileNotFoundError(
                f"Source directory not found at expected path: {src_path}"
            )
        # Use str() for compatibility across Python versions.
        sys.path.insert(0, str(src_path))
        return project_root
    except FileNotFoundError as e:
        print(f"Error: Could not dynamically set python path. {e}", file=sys.stderr)
        print(
            "Please ensure the script is run from within the project structure.",
            file=sys.stderr,
        )
        sys.exit(1)


# Execute path setup immediately. This must be done before the 'cortex' import.
PROJECT_ROOT = setup_sys_path()

# Now, we can safely import from the cortex package
from cortex.config import models
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
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, val = line.split("=", 1)
                env_vars[key.strip()] = val.strip()
    return env_vars


def parse_models_via_introspection() -> dict[str, dict[str, Any]]:
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

    original_env = models._env
    original_env_list = models._env_list
    models._env = mock_env
    models._env_list = lambda key, default="": mock_env(key, default, list)

    try:
        config_classes: list[type[BaseModel]] = [
            obj
            for obj in vars(models).values()
            if inspect.isclass(obj)
            and issubclass(obj, BaseModel)
            and obj is not BaseModel
        ]
        for model_class in config_classes:
            try:
                model_class.model_validate({})
            except Exception:
                continue
    finally:
        models._env = original_env
        models._env_list = original_env_list
    return definitions


def _normalize_value(val: str) -> str:
    """Normalize a value for comparison (lowercase, strip quotes)."""
    return val.lower().strip().strip('"').strip("'")


def _determine_status(
    model_def: dict | None, env_val: str | None, code_default: str
) -> str:
    """Determine the status of a config key comparison."""
    if model_def and env_val is not None:
        if _normalize_value(str(code_default)) == _normalize_value(env_val):
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
    if (
        status == "ENV_ONLY"
        and key.isupper()
        and ("OUTLOOKCORTEX_" in key or "EMAILOPS_" in key)
    ):
        return True
    return False


def _get_env_value(env_vars: dict[str, str], key: str) -> str | None:
    """Get environment value, checking for standard prefixes first."""
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
    sensitive_patterns = ["key", "secret", "password", "token", "url", "credentials"]
    return any(pattern in key_lower for pattern in sensitive_patterns)


def main():
    """Main execution function."""
    env_vars = parse_env_file(PROJECT_ROOT / ".env")
    model_defs = parse_models_via_introspection()

    print(f"{'KEY':<35} | {'CODE DEFAULT':<40} | {'ENV VALUE':<40} | {'STATUS':<15}")
    print("-" * 135)

    all_keys = sorted(set(env_vars.keys()) | set(model_defs.keys()))
    app_keys = {
        k
        for k in all_keys
        if k in model_defs or "OUTLOOKCORTEX_" in k or "EMAILOPS_" in k
    }

    for key in sorted(list(app_keys)):
        model_def = model_defs.get(key)
        env_val = _get_env_value(env_vars, key)
        code_default_val = model_def.get("default") if model_def else "N/A"
        code_default = str(code_default_val)
        env_display = str(env_val) if env_val is not None else "N/A"

        if _is_sensitive(key):
            if code_default_val not in ("N/A", None, ""):
                code_default = "***REDACTED***"
            if env_val is not None and env_val != "":
                env_display = "***REDACTED***"

        status = _determine_status(model_def, env_val, code_default)
        if _should_print_entry(status, key):
            print(f"{key:<35} | {code_default:<40} | {env_display:<40} | {status:<15}")


if __name__ == "__main__":
    main()
