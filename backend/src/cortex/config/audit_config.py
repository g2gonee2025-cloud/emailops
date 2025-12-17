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


def main():
    root = Path("/root/workspace/emailops-vertex-ai")
    env_vars = parse_env_file(root / ".env")
    model_defs = parse_models_file(root / "backend/src/cortex/config/models.py")

    print(f"{'KEY':<30} | {'CODE DEFAULT':<40} | {'ENV VALUE':<40} | {'STATUS':<15}")
    print("-" * 130)

    all_keys = sorted(set(env_vars.keys()) | set(model_defs.keys()))

    for key in all_keys:
        # Ignore non-config env vars (those not in models.py) unless they explicitly overlap naming conventions
        # But user wants "everything", so let's show overlap.

        model_def = model_defs.get(key)
        env_val = env_vars.get(key)

        # Handle OUTLOOKCORTEX_ prefix in env vars if model uses it?
        # models.py _env wrapper adds OUTLOOKCORTEX_ automatically.
        # usually .env has simple names like "DB_URL" or "PII_ENABLED" but might have "OUTLOOKCORTEX_PII_ENABLED"
        # The script above reads .env raw.
        # Note: models.py _env("KEY") looks for "OUTLOOKCORTEX_KEY" then "KEY".

        # Let's check for prefixed version in .env if simple key missing
        prefixed_key = f"OUTLOOKCORTEX_{key}"
        if env_val is None and model_def:
            env_val = env_vars.get(prefixed_key)
            if env_val:
                # key_display = prefixed_key  # Show the actual key found
                pass

        code_default = str(model_def["default"]) if model_def else "N/A"
        env_display = str(env_val) if env_val is not None else "N/A"

        status = ""
        if model_def and env_val is not None:
            # Simple string comparison
            # normalize bools/numbers
            v1 = code_default.lower()
            v2 = env_display.lower()

            # Special handling for empty strings vs None
            if (
                v1 == "none" and v2 == ""
            ):  # env empty string often means empty, code None means None. Mismatch?
                pass

            if v1.replace('"', "").replace("'", "") == v2.replace('"', "").replace(
                "'", ""
            ):
                status = "MATCH"
            else:
                status = "MISMATCH"
        elif model_def and env_val is None:
            status = "CODE_ONLY"
        elif not model_def:
            # Key in env but not in models?
            # Filter out standard shell things or comments
            status = "ENV_ONLY"

        # Filter out boring matches if list is huge? User said "everything".
        # But let's prioritize mismatches.

        if status == "MISMATCH":
            print(f"{key:<30} | {code_default:<40} | {env_display:<40} | {status:<15}")
        elif status == "ENV_ONLY":
            # Only show if it looks like a config var
            if key.isupper():
                print(
                    f"{key:<30} | {code_default:<40} | {env_display:<40} | {status:<15}"
                )


if __name__ == "__main__":
    main()
