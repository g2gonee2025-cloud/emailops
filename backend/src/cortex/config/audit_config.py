import re
from pathlib import Path


def parse_env_file(path):
    env_vars = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, val = line.split("=", 1)
                env_vars[key.strip()] = val.strip()
    return env_vars


def parse_models_file(path):
    definitions = {}
    with open(path, "r") as f:
        content = f.read()

    # Regex to find _env("KEY", default) calls
    # Matches: _env("KEY", "default") or _env("KEY", 123) or _env("KEY", True, bool)
    # This is rough but should catch most standard usages in this file
    pattern = r'_env\(\s*"([^"]+)"\s*,\s*([^,)]+)(?:,\s*([^)]+))?\)'

    for match in re.finditer(pattern, content):
        key = match.group(1)
        default_raw = match.group(2).strip()
        type_hint = match.group(3).strip() if match.group(3) else None

        # Clean up quotes from strings
        if default_raw.startswith('"') and default_raw.endswith('"'):
            default_val = default_raw[1:-1]
        elif default_raw.startswith("'") and default_raw.endswith("'"):
            default_val = default_raw[1:-1]
        else:
            default_val = default_raw

        definitions[key] = {"default": default_val, "type": type_hint}
    return definitions


def _normalize_value(val: str) -> str:
    """Normalize a value for comparison (lowercase, strip quotes)."""
    return val.lower().replace('"', "").replace("'", "")


def _determine_status(
    model_def: dict | None, env_val: str | None, code_default: str
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
    return ""


def _should_print_entry(status: str, key: str) -> bool:
    """Determine if an entry should be printed."""
    if status == "MISMATCH":
        return True
    if status == "ENV_ONLY" and key.isupper():
        return True
    return False


def _get_env_value(env_vars: dict, key: str, model_def: dict | None) -> str | None:
    """Get environment value, checking prefixed version if needed."""
    env_val = env_vars.get(key)
    if env_val is None and model_def:
        prefixed_key = f"OUTLOOKCORTEX_{key}"
        env_val = env_vars.get(prefixed_key)
    return env_val


def main():
    root = Path("/root/workspace/emailops-vertex-ai")
    env_vars = parse_env_file(root / ".env")
    model_defs = parse_models_file(root / "backend/src/cortex/config/models.py")

    print(f"{'KEY':<30} | {'CODE DEFAULT':<40} | {'ENV VALUE':<40} | {'STATUS':<15}")
    print("-" * 130)

    all_keys = sorted(set(env_vars.keys()) | set(model_defs.keys()))

    for key in all_keys:
        model_def = model_defs.get(key)
        env_val = _get_env_value(env_vars, key, model_def)

        code_default = str(model_def["default"]) if model_def else "N/A"
        env_display = str(env_val) if env_val is not None else "N/A"

        status = _determine_status(model_def, env_val, code_default)

        if _should_print_entry(status, key):
            print(f"{key:<30} | {code_default:<40} | {env_display:<40} | {status:<15}")


if __name__ == "__main__":
    main()
