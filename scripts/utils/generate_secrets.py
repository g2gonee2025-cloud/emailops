import base64
import re
import sys
from pathlib import Path
from textwrap import dedent
from typing import Union


def load_env_file(filepath: str | Path = ".env") -> dict[str, str]:
    """
    Load .env file, handling comments, quotes, and variable expansions.
    This avoids dependencies like python-dotenv for simple use cases.
    """
    env_vars: dict[str, str] = {}
    env_path = Path(filepath)

    if not env_path.exists():
        raise FileNotFoundError(f"Error: {filepath} not found")

    try:
        with env_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                # Regex to parse 'KEY=VALUE' pairs, handling optional quotes
                match = re.match(r"^\s*([\w.-]+)\s*=\s*(.*?)?\s*$", line)
                if match:
                    key, val = match.groups()
                    # Remove surrounding quotes (single or double)
                    if val and val[0] == val[-1] and val[0] in ('"', "'"):
                        val = val[1:-1]
                    env_vars[key] = val
    except (OSError, UnicodeDecodeError) as e:
        raise ValueError(f"Error reading or parsing {filepath}: {e}") from e

    return env_vars


def b64_encode(value: str) -> str:
    if not value:
        return ""
    return base64.b64encode(value.encode("utf-8")).decode("utf-8")


def generate_secrets_yaml() -> None:
    try:
        env = load_env_file()
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: Could not load .env file. {e}", file=sys.stderr)
        sys.exit(1)

    # Keys to extract from .env and put into k8s secret
    # Mapping .env key -> K8s secret key
    keys_map = {
        "OUTLOOKCORTEX_DB_URL": "OUTLOOKCORTEX_DB_URL",
        "REDIS_URL": "REDIS_URL",
        "S3_ACCESS_KEY": "S3_ACCESS_KEY",
        "S3_SECRET_KEY": "S3_SECRET_KEY",
        "DO_LLM_API_KEY": "DO_LLM_API_KEY",
        "DO_TOKEN": "DO_TOKEN",
    }

    encoded_data = {}
    for env_key, secret_key in keys_map.items():
        val = env.get(env_key) or ""
        if not val:
            print(f"Warning: {env_key} not found in .env, value will be empty.")
        encoded_data[secret_key] = b64_encode(val)

    # Dynamically generate the 'data' block
    data_yaml = "\n".join(
        [f"    {key}: {value}" for key, value in encoded_data.items()]
    )

    yaml_content = dedent(
        f"""
            apiVersion: v1
            kind: Secret
            metadata:
              name: emailops-secrets
              namespace: emailops
            type: Opaque
            data:
            {data_yaml}
            """
    ).strip()

    # SECURITY: Write to a user-level config path, not a project path.
    output_path = Path.home() / ".config" / "cortex" / "secrets.yaml"
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            f.write(yaml_content)
        # SECURITY: Restrict file permissions to owner-only.
        output_path.chmod(0o600)
    except (OSError, PermissionError) as e:
        print(
            f"Error: Could not write to {output_path}. Check permissions. {e}",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Successfully generated secrets at: {output_path}")


if __name__ == "__main__":
    generate_secrets_yaml()
