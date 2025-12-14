import base64
import sys
from pathlib import Path
from textwrap import dedent


def load_env_file(filepath: str | Path = ".env"):
    """Load .env file manually to avoid dependency"""
    env_vars: dict[str, str] = {}
    env_path = Path(filepath)
    if not env_path.exists():
        print(f"Error: {filepath} not found")
        sys.exit(1)

    with env_path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, val = line.split("=", 1)
                env_vars[key.strip()] = val.strip()
    return env_vars


def b64_encode(value):
    if not value:
        return ""
    return base64.b64encode(value.encode("utf-8")).decode("utf-8")


def generate_secrets_yaml():
    env = load_env_file()

    # Keys to extract from .env and put into k8s secret
    # Mapping .env key -> K8s secret key
    keys_map = {
        "OUTLOOKCORTEX_DB_URL": "OUTLOOKCORTEX_DB_URL",
        "REDIS_URL": "REDIS_URL",
        "S3_ACCESS_KEY": "S3_ACCESS_KEY",
        "S3_SECRET_KEY": "S3_SECRET_KEY",
        "MODEL_ACCESS_KEY": "MODEL_ACCESS_KEY",
        "DO_TOKEN": "DO_TOKEN",
    }

    encoded_data = {}
    for env_key, secret_key in keys_map.items():
        val = env.get(env_key, "")
        if not val:
            print(f"Warning: {env_key} not found in .env")
        encoded_data[secret_key] = b64_encode(val)

        yaml_content = dedent(
            f"""
                apiVersion: v1
                kind: Secret
                metadata:
                    name: emailops-secrets
                    namespace: emailops
                type: Opaque
                data:
                    # Database connection
                    OUTLOOKCORTEX_DB_URL: {encoded_data['OUTLOOKCORTEX_DB_URL']}

                    # Valkey cache connection
                    REDIS_URL: {encoded_data['REDIS_URL']}

                    # S3/Spaces Access Keys
                    S3_ACCESS_KEY: {encoded_data['S3_ACCESS_KEY']}
                    S3_SECRET_KEY: {encoded_data['S3_SECRET_KEY']}

                    # AI Provider Keys
                    MODEL_ACCESS_KEY: {encoded_data['MODEL_ACCESS_KEY']}
                    DO_TOKEN: {encoded_data['DO_TOKEN']}
                """
        )

        output_path = Path("k8s/secrets_live.yaml")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as f:
            f.write(yaml_content)

        print(
            f"Successfully generated {output_path} with base64 encoded secrets from .env"
        )


if __name__ == "__main__":
    generate_secrets_yaml()
