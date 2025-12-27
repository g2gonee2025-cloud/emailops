from pathlib import Path

root_dir = Path(__file__).resolve().parents[2]
from pathlib import Path

from cortex.config.loader import get_config


def _safe_get(obj, *attrs):
    cur = obj
    for a in attrs:
        if cur is None:
            return None
        try:
            cur = getattr(cur, a)
        except AttributeError:
            if isinstance(cur, dict):
                cur = cur.get(a)
            else:
                return None
    return cur


def verify_config():
    config = get_config()
    core_env = _safe_get(config, "core", "env")
    db_url = _safe_get(config, "database", "url")
    masked_db = db_url.split("@")[-1] if isinstance(db_url, str) and db_url else None
    s3_bucket = _safe_get(config, "storage", "bucket_raw")
    out_dim = _safe_get(config, "embedding", "output_dimensionality")
    print("Core Env:", core_env if core_env is not None else "N/A")
    print("DB URL:", masked_db if masked_db is not None else "N/A")  # Masked
    print("S3 Bucket:", s3_bucket if s3_bucket is not None else "N/A")
    print("Output Dimension:", out_dim if out_dim is not None else "N/A")
    print(
        "Redis URL:",
        (
            config.redis.url.split("@")[-1]
            if "@" in config.redis.url
            else config.redis.url
        ),
    )
    print("Chunk Size:", config.processing.chunk_size)
    print("Max Retries:", config.retry.max_retries)


if __name__ == "__main__":
    verify_config()
