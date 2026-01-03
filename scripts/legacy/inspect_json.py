import json
import sys
from pathlib import Path
from typing import Any

# Add the backend src directory to the Python path
_BACKEND_SRC = Path(__file__).resolve().parents[2] / "backend" / "src"
sys.path.insert(0, str(_BACKEND_SRC))

from cortex.common.exceptions import ConfigurationError
from cortex.config.loader import get_config
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

EMPTY_JSON_OBJECT = "{}"
_SENSITIVE_KEY_MARKERS = (
    "key",
    "secret",
    "token",
    "password",
    "credential",
    "auth",
    "jwt",
    "private",
    "cert",
    "api",
    "email",
    "name",
    "subject",
    "from",
    "to",
    "cc",
    "bcc",
)


def _redact_value(value: Any) -> Any:
    """Return a redacted placeholder for a value."""
    if value is None or value == "":
        return value
    return "***REDACTED***"


def redact_sensitive_data(data: Any) -> Any:
    """Recursively redact sensitive information from a data structure."""
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except json.JSONDecodeError:
            return data  # Not a JSON string, return as is.

    if isinstance(data, dict):
        redacted: dict[str, Any] = {}
        for key, value in data.items():
            if isinstance(key, str) and any(
                marker in key.lower() for marker in _SENSITIVE_KEY_MARKERS
            ):
                redacted[key] = _redact_value(value)
            else:
                redacted[key] = redact_sensitive_data(value)
        return redacted
    if isinstance(data, list):
        return [redact_sensitive_data(item) for item in data]
    return data


def inspect_metadata() -> None:
    """Inspect metadata using the application configuration.

    Reads the application configuration to access the database URL and performs metadata-related inspection.
    """
    try:
        config = get_config()
        db_url = config.database.url
        if not db_url:
            print("Database URL is not configured.")
            return

        engine = create_engine(db_url)
        with engine.connect() as conn:
            # Get samples from threads metadata where it's not empty/null
            sql = text(
                """
                SELECT metadata
                FROM threads
                WHERE metadata IS NOT NULL AND CAST(metadata AS TEXT) != :empty_json
                LIMIT 5
            """
            )
            result = conn.execute(sql, {"empty_json": EMPTY_JSON_OBJECT}).fetchall()

            print("----- Threads Metadata Samples -----")
            if not result:
                print("No non-empty metadata found in threads.")
            else:
                for row in result:
                    redacted_metadata = redact_sensitive_data(row[0])
                    print(json.dumps(redacted_metadata, indent=2))

            # Also check messages metadata as it might be richer
            print("\n----- Messages Metadata Samples -----")
            sql_msg = text(
                """
                SELECT metadata
                FROM messages
                WHERE metadata IS NOT NULL AND CAST(metadata AS TEXT) != :empty_json
                LIMIT 5
            """
            )
            result_msg = conn.execute(
                sql_msg, {"empty_json": EMPTY_JSON_OBJECT}
            ).fetchall()

            if not result_msg:
                print("No non-empty metadata found in messages.")
            else:
                for row in result_msg:
                    redacted_metadata = redact_sensitive_data(row[0])
                    print(json.dumps(redacted_metadata, indent=2))

    except ConfigurationError as e:
        print(f"Configuration Error: {e}", file=sys.stderr)
        sys.exit(1)
    except SQLAlchemyError as e:
        print(f"Database Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    inspect_metadata()
