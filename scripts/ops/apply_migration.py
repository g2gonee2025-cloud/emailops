import os
import sys
from pathlib import Path

import alembic.config

# Add backend/src and backend/migrations to path
sys.path.append(str(Path(__file__).resolve().parents[2] / "backend" / "src"))
# sys.path.append(os.path.join(os.getcwd(), 'backend', 'migrations'))

from cortex.config.loader import get_config


def apply():
    config = get_config()
    os.environ["OUTLOOKCORTEX_DB_URL"] = config.database.url
    os.environ["EMBED_DIM"] = str(config.embedding.output_dimensionality)

    print(
        f"Applying migrations to: {config.database.url.split('@')[-1]}"
    )  # redact password

    # We must run this from the backend directory so alembic.ini is found
    # os.path.join(os.getcwd(), "backend", "migrations")

    # Alembic arguments: [prog, command, revision]
    # But wait, alembic.config.main() parses args.
    # It expects to be run from the directory containing alembic.ini usually.

    original_cwd = Path.cwd()
    os.chdir(original_cwd / "backend" / "migrations")

    try:
        alembic.config.main(argv=["upgrade", "head"])
    except Exception as e:
        print(f"Migration failed: {e}")
    finally:
        os.chdir(original_cwd)


if __name__ == "__main__":
    apply()
