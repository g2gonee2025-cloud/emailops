import sys
from pathlib import Path
from urllib.parse import urlparse

def find_project_root(marker="pyproject.toml"):
    """Find the project root by searching upwards for a marker file."""
    current_dir = Path(__file__).resolve()
    while current_dir != current_dir.parent:
        if (current_dir / marker).exists():
            return current_dir
        current_dir = current_dir.parent
    raise FileNotFoundError(f"Project root with marker '{marker}' not found.")

# Add the project root to the Python path to allow for absolute imports
try:
    project_root = find_project_root()
    sys.path.append(str(project_root / "backend" / "src"))
except FileNotFoundError as e:
    print(f"Error: Could not find project root. {e}", file=sys.stderr)
    sys.exit(1)


from cortex.config.loader import get_config


def _safe_get(obj, *attrs):
    """
    Safely retrieve a nested attribute or key.
    Prioritizes dictionary key lookup over attribute access to avoid conflicts.
    """
    cur = obj
    for a in attrs:
        if cur is None:
            return None

        # LOGIC_ERROR FIX: Prioritize dictionary lookup to prevent issues where a
        # key name conflicts with an object's method/attribute name (e.g., 'items').
        if isinstance(cur, dict):
            cur = cur.get(a)
        else:
            # EXCEPTION_HANDLING FIX: By checking for dict type first, this now only
            # attempts attribute access on non-dict objects. The broad exception is
            # now less likely to mask unrelated errors.
            try:
                cur = getattr(cur, a)
            except AttributeError:
                return None
    return cur


def _mask_url(url: str) -> str:
    """Masks credentials in a URL, returning a safe version to print."""
    if not isinstance(url, str) or not url:
        return None
    try:
        parsed = urlparse(url)

        # Rebuild URL with masked password and no query parameters for security
        if parsed.password:
            netloc_parts = []
            if parsed.username:
                netloc_parts.append(parsed.username)

            # Handle URLs with password but no username, e.g., redis://:pass@...
            if not parsed.username and parsed.netloc.startswith(':'):
                netloc_parts.append('')

            netloc_parts.append(":*****")

            if parsed.hostname:
                netloc_parts.append(f"@{parsed.hostname}")
                if parsed.port:
                    netloc_parts.append(f":{parsed.port}")

            new_netloc = "".join(netloc_parts)

            # Strip query parameters for security when a password is present
            return parsed._replace(netloc=new_netloc, query="").geturl()

        # If no password, but a query string exists, mask the query string
        if parsed.query:
            return parsed._replace(query="[MASKED]").geturl()

        return url

    except Exception:
        # Fallback for malformed URLs or parsing errors
        return "[URL MASKING FAILED]"


def verify_config():
    try:
        config = get_config()
        core_env = _safe_get(config, "core", "env")
        db_url = _safe_get(config, "database", "url")
        redis_url = _safe_get(config, "redis", "url")
        s3_bucket = _safe_get(config, "storage", "bucket_raw")
        out_dim = _safe_get(config, "embedding", "output_dimensionality")
        chunk_size = _safe_get(config, "processing", "chunk_size")
        max_retries = _safe_get(config, "retry", "max_retries")

        print("Core Env:", core_env if core_env is not None else "N/A")
        print("DB URL:", _mask_url(db_url) or "N/A")
        print("Redis URL:", _mask_url(redis_url) or "N/A")
        print("S3 Bucket:", s3_bucket if s3_bucket is not None else "N/A")
        print("Output Dimension:", out_dim if out_dim is not None else "N/A")
        print("Chunk Size:", chunk_size if chunk_size is not None else "N/A")
        print("Max Retries:", max_retries if max_retries is not None else "N/A")

    except Exception as e:
        print(f"Error: Failed to verify configuration. Reason: {e}")
        sys.exit(1)


if __name__ == "__main__":
    verify_config()
