import sys
from pathlib import Path

sys.path.append(str(Path("backend/src").resolve()))
from cortex.config.loader import get_config


def verify_config():
    config = get_config()
    print("Core Env:", config.core.env)
    print("DB URL:", config.database.url.split("@")[-1])  # Masked
    print("S3 Bucket:", config.storage.bucket_raw)
    print("Output Dimension:", config.embedding.output_dimensionality)


if __name__ == "__main__":
    verify_config()
