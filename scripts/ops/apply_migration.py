import logging
import os
import sys
from pathlib import Path

import alembic.config

def find_project_root(marker: str = "pyproject.toml") -> Path:
    """Locate the project root by searching for a marker file."""
    current_path = Path(__file__).resolve()
    for parent in current_path.parents:
        if (parent / marker).exists():
            return parent
    raise FileNotFoundError(f"Could not find project root marker: {marker}")


def main() -> None:
    """Configures environment and runs alembic migrations."""
    logger = logging.getLogger(__name__)

    try:
        # --- Robust Path Setup ---
        project_root = find_project_root()
        backend_src_path = project_root / "backend" / "src"
        migrations_dir = project_root / "backend" / "migrations"

        if str(backend_src_path) not in sys.path:
            sys.path.append(str(backend_src_path))

        from cortex.config.loader import get_config

        # --- Configuration Loading and Validation ---
        config = get_config()

        db_url = getattr(config.database, "url", None)
        embed_dim = getattr(config.embedding, "output_dimensionality", None)

        if not db_url:
            logger.error("Database URL is not configured.")
            sys.exit(1)
        if embed_dim is None: # Can be 0, so check for None
            logger.error("Embedding output dimensionality is not configured.")
            sys.exit(1)

        os.environ["OUTLOOKCORTEX_DB_URL"] = str(db_url)
        os.environ["EMBED_DIM"] = str(embed_dim)

        logger.info("Applying migrations to the configured database.")

        # --- Run Migrations ---
        original_cwd = Path.cwd()
        try:
            os.chdir(migrations_dir)
            alembic.config.main(argv=["--raiseerr", "upgrade", "head"])
            logger.info("Migrations applied successfully.")
        except Exception:
            logger.exception("Migration failed.")
            sys.exit(1) # Propagate failure
        finally:
            os.chdir(original_cwd)

    except FileNotFoundError as e:
        logger.error(f"Setup failed: {e}")
        sys.exit(1)
    except Exception:
        logger.exception("An unexpected error occurred during migration setup.")
        sys.exit(1)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    main()
