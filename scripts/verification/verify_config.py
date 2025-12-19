import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parents[2]
from pathlib import Path  # noqa: E402

sys.path.append(str(Path("backend/src").resolve()))
from cortex.config.loader import get_config  # noqa: E402


def verify_config():
    config = get_config()
    print("Core Env:", config.core.env)
    print("DB URL:", config.database.url.split("@")[-1])  # Masked
    print("S3 Bucket:", config.storage.bucket_raw)
    print("Output Dimension:", config.embedding.output_dimensionality)
    print(
        "Redis URL:",
        config.redis.url.split("@")[-1]
        if "@" in config.redis.url
        else config.redis.url,
    )
    print("Chunk Size:", config.processing.chunk_size)
    print("Max Retries:", config.retry.max_retries)


if __name__ == "__main__":
    verify_config()
