import os
import secrets
import string
import sys
import time
from pathlib import Path

import requests
from dotenv import set_key

# --- Configuration ---
SONAR_URL = "https://localhost:9000"
SONAR_ADMIN_USER = os.environ.get("SONAR_ADMIN_USER")
SONAR_ADMIN_PASSWORD = os.environ.get("SONAR_ADMIN_PASSWORD")
REQUEST_TIMEOUT = 10  # seconds

# Environment-aware SSL verification (S4830 fix)
# Enable SSL verification by default, only disable in local dev with explicit flag
DEV_MODE = os.environ.get("EMAILOPS_DEV_MODE", "false").lower() == "true"
DISABLE_SSL_VERIFY = os.environ.get("SONAR_DISABLE_SSL_VERIFY", "false").lower() == "true"
VERIFY_SSL = not (DEV_MODE and DISABLE_SSL_VERIFY)

if not VERIFY_SSL:
    import warnings
    warnings.warn(
        "SSL certificate verification is disabled. This should only be used in local development.",
        category=SecurityWarning,
        stacklevel=2
    )


def generate_random_string(length=8):
    """Generates a random alphanumeric string."""
    alphabet = string.ascii_letters + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(length))


def find_project_root():
    """Dynamically finds the project root by looking for a known file/directory."""
    current_path = Path(__file__).resolve().parent
    while current_path != current_path.parent:
        if (current_path / "pyproject.toml").exists() and (
            current_path / "backend"
        ).is_dir():
            return current_path
        current_path = current_path.parent
    raise FileNotFoundError("Could not find the project root.")


def main():
    """
    Generates a SonarQube token and updates the .env file.
    
    Security fixes applied:
    - S4830: SSL certificate validation enabled by default
    - Environment-aware SSL verification with explicit dev mode flag
    """
    if not SONAR_ADMIN_USER or not SONAR_ADMIN_PASSWORD:
        print(
            "Error: SONAR_ADMIN_USER and SONAR_ADMIN_PASSWORD environment variables must be set."
        )
        sys.exit(1)

    timestamp = int(time.time())
    random_part = generate_random_string()
    token_name = f"agent-token-{timestamp}-{random_part}"

    try:
        print(f"Generating token '{token_name}'...")
        print(f"SSL certificate verification: {'ENABLED' if VERIFY_SSL else 'DISABLED (dev mode)'}")
        
        r = requests.post(
            f"{SONAR_URL}/api/user_tokens/generate",
            params={"name": token_name},
            auth=(SONAR_ADMIN_USER, SONAR_ADMIN_PASSWORD),
            timeout=REQUEST_TIMEOUT,
            verify=VERIFY_SSL,  # S4830: SSL verification enabled by default
        )

        if r.status_code != 200:
            print(
                f"Error generating token. Server responded with status code: {r.status_code}"
            )
            # Do not print r.text to avoid leaking sensitive info
            sys.exit(1)

        token = r.json()["token"]
        print("Token generated successfully.")

        # Update .env file
        project_root = find_project_root()
        env_path = project_root / ".env"

        print(f"Updating {env_path}...")
        set_key(env_path, "SONAR_TOKEN", token, quote_mode="always")
        set_key(env_path, "SONAR_HOST_URL", SONAR_URL, quote_mode="always")

        # Set restrictive file permissions
        os.chmod(env_path, 0o600)

        print(f"Successfully updated {env_path}")

    except requests.exceptions.Timeout:
        print(f"Error: Request to SonarQube timed out after {REQUEST_TIMEOUT} seconds.")
        sys.exit(1)
    except requests.exceptions.RequestException as e:
        print(f"Error connecting to SonarQube: {e}")
        sys.exit(1)
    except (OSError, ValueError, KeyError) as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
