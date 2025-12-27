import os
import sys
from pathlib import Path

import alembic.config

# Add backend/src and backend/migrations to path
sys.path.append(str(Path(__file__).resolve().parents[2] / "backend" / "src"))
# sys.path.append(os.path.join(os.getcwd(), 'backend', 'migrations'))

from cortex.config.loader import get_config


def apply():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("apply_migration")

    try:
        config = get_config()
        os.environ["OUTLOOKCORTEX_DB_URL"] = config.database.url
        os.environ["EMBED_DIM"] = str(config.embedding.output_dimensionality)

        logger.info("Applying migrations to: %s", config.database.url.split("@")[-1])

        original_cwd = Path.cwd()
        os.chdir(original_cwd / "backend" / "migrations")

        try:
            alembic.config.main(argv=["upgrade", "head"])
        except Exception as e:
            logger.error("Migration failed: %s", e)
        finally:
            os.chdir(original_cwd)

    except Exception as e:
        logger.error("Failed to setup migration: %s", e)


if __name__ == "__main__":
    apply()
