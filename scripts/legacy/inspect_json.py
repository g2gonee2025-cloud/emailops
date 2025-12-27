# Securely import get_config without modifying sys.path
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

from sqlalchemy import create_engine, text

_BACKEND_SRC = Path(__file__).resolve().parents[2] / "backend" / "src"
_LOADER_PATH = _BACKEND_SRC / "cortex" / "config" / "loader.py"
_spec = spec_from_file_location("cortex.config.loader", _LOADER_PATH)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Could not load cortex.config.loader from {_LOADER_PATH}")
_module = module_from_spec(_spec)
_spec.loader.exec_module(_module)
get_config = _module.get_config


def inspect_metadata() -> None:
    """Inspect metadata using the application configuration.

    Reads the application configuration to access the database URL and performs metadata-related inspection.
    """
    try:
        config = get_config()
        db_url = config.database.url
        if "sslmode" not in db_url:
            db_url += ("&" if "?" in db_url else "?") + "sslmode=require"

        engine = create_engine(db_url)
        with engine.connect() as conn:
            # Get samples from threads metadata where it's not empty/null
            EMPTY_JSON_OBJECT = "{}"
            sql = text(
                f"""
                SELECT metadata
                FROM threads
                WHERE metadata IS NOT NULL AND metadata::text != '{EMPTY_JSON_OBJECT}'
                LIMIT 5
            """
            )
            result = conn.execute(sql).fetchall()

            print("----- Threads Metadata Samples -----")
            if not result:
                print("No non-empty metadata found in threads.")
            else:
                for row in result:
                    print(row.metadata)

            # Also check messages metadata as it might be richer
            print("\n----- Messages Metadata Samples -----")
            sql_msg = text(
                """
                SELECT metadata
                FROM messages
                WHERE metadata IS NOT NULL AND metadata::text != '{}'
                LIMIT 5
            """
            )
            result_msg = conn.execute(sql_msg).fetchall()

            if not result_msg:
                print("No non-empty metadata found in messages.")
            else:
                for row in result_msg:
                    print(row.metadata)

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    inspect_metadata()
