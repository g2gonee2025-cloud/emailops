"""
Doctor Command Module.

Implements system health checks for the Cortex Doctor CLI.
"""

import os

import structlog
from cortex.config.loader import get_config

logger = structlog.get_logger(__name__)


class CortexDoctor:
    """
    Diagnoses system health by checking connectivity to critical services.
    """

    def __init__(self):
        self.config = get_config()

    def check_env(self) -> bool:
        """Check environment variables."""
        required_keys = ["OUTLOOKCORTEX_DB_URL", "OUTLOOKCORTEX_S3_ACCESS_KEY"]

        missing = [key for key in required_keys if not os.environ.get(key)]

        if missing:
            # Also check if config loaded them (fallback)
            if self.config:
                # If config exists, we assume it loaded them, but let's verify specifics
                if "OUTLOOKCORTEX_DB_URL" in missing and self.config.database.url:
                    missing.remove("OUTLOOKCORTEX_DB_URL")
                if "OUTLOOKCORTEX_S3_ACCESS_KEY" in missing and self.config.storage.access_key:
                    missing.remove("OUTLOOKCORTEX_S3_ACCESS_KEY")

        if missing:
            logger.error("env_check_failed", missing_keys=missing)
            return False

        logger.info("env_check_passed", status="OK")
        return True

    def check_db(self) -> bool:
        """Check Database Connectivity."""
        try:
            from cortex.db.session import engine
            from sqlalchemy import text

            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))

            logger.info("db_check", status="OK", url="[MASKED]")
            return True
        except Exception as e:
            logger.error("db_check_failed", error=str(e))
            return False

    def check_s3(self) -> bool:
        """Check Object Storage Connectivity."""
        try:
            from cortex.ingestion.s3_source import S3SourceHandler

            with S3SourceHandler() as s3_handler:
                s3_handler.client.list_objects_v2(
                    Bucket=s3_handler.bucket, MaxKeys=1
                )
                bucket_used = s3_handler.bucket
            logger.info("s3_check", status="OK", bucket=bucket_used)
            return True
        except Exception as e:
            logger.error("s3_check_failed", error=str(e))
            return False

    def run_all(self):
        """Run all checks."""
        print("Running Cortex Doctor...")
        checks = [
            ("Environment", self.check_env),
            ("Database", self.check_db),
            ("Object Storage", self.check_s3),
        ]

        all_passed = True
        for name, check in checks:
            print(f"Checking {name}...", end=" ")
            if check():
                print("✅ OK")
            else:
                print("❌ FAILED")
                all_passed = False

        if all_passed:
            print("\nSystem is HEALTHY.")
            return True
        else:
            print("\nSystem has ISSUES.")
            return False
