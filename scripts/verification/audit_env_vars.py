import re
from pathlib import Path


def parse_env_file(env_path: Path) -> set[str]:
    """Extract keys from .env file."""
    keys = set()
    if not env_path.exists():
        print(f"Warning: {env_path} not found.")
        return keys

    with env_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Handle export VAR=VAL
            line = re.sub(r"^export\s+", "", line)
            if "=" in line:
                key = line.split("=", 1)[0].strip()
                keys.add(key)
    return keys


def scan_codebase_for_env_usage(root_dir: Path) -> dict[str, list[str]]:
    """
    Scan python files for env var usage.
    Returns dict: var_name -> list of file paths
    """
    usage = {}

    # Regex patterns
    # _env("KEY", ...)
    # os.getenv("KEY", ...)
    # os.environ.get("KEY", ...)
    # os.environ["KEY"]
    patterns = [
        re.compile(r'_env\(\s*["\']([A-Z0-9_]+)["\']'),
        re.compile(r'os\.getenv\(\s*["\']([A-Z0-9_]+)["\']'),
        re.compile(r'os\.environ\.get\(\s*["\']([A-Z0-9_]+)["\']'),
        re.compile(r'os\.environ\[\s*["\']([A-Z0-9_]+)["\']'),
    ]

    for py_file in root_dir.rglob("*.py"):
        try:
            content = py_file.read_text(encoding="utf-8")
            for pattern in patterns:
                for match in pattern.findall(content):
                    if match not in usage:
                        usage[match] = []
                    usage[match].append(str(py_file.relative_to(root_dir)))
        except Exception as e:
            print(f"Error reading {py_file}: {e}")

    return usage


def main():
    root_dir = Path.cwd()
    env_path = root_dir / ".env"
    src_dir = root_dir / "backend" / "src"

    defined_vars = parse_env_file(env_path)
    used_vars = scan_codebase_for_env_usage(src_dir)

    print("=== Environment Variable Audit ===\n")

    # 1. Variables in .env but NOT found in codebase
    # Note: Some might be used by docker-compose or infra scripts, so this is just a warning.
    unused = sorted([v for v in defined_vars if v not in used_vars])
    print(f"--- Defined in .env but NOT found in backend/src ({len(unused)}) ---")
    for v in unused:
        # Ignore canonical prefixes if they are just prefixes
        if not v.startswith("OUTLOOKCORTEX_") and not v.startswith("EMAILOPS_"):
            print(f"  {v}")
        else:
            # Check if the suffix is used (cleaner check)
            suffix = v.replace("OUTLOOKCORTEX_", "").replace("EMAILOPS_", "")
            if suffix not in used_vars and v not in used_vars:
                print(f"  {v}")

    print("\n")

    # 2. Variables used in codebase but NOT in .env
    # This is critical for missing config.
    missing = sorted([v for v in used_vars if v not in defined_vars])

    # Filter out known non-env vars or dynamic/system ones if needed
    confirmed_missing = []
    for v in missing:
        # Check if legacy or prefix variants exist
        has_legacy = f"EMAILOPS_{v}" in defined_vars
        has_canonical = f"OUTLOOKCORTEX_{v}" in defined_vars

        if not (has_legacy or has_canonical):
            confirmed_missing.append(v)

    print(f"--- Used in code but MISSING from .env ({len(confirmed_missing)}) ---")
    for v in confirmed_missing:
        files = ", ".join(used_vars[v][:2])  # show first 2 files
        if len(used_vars[v]) > 2:
            files += "..."
        print(f"  {v} (used in: {files})")


if __name__ == "__main__":
    main()
