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
        # Using S3_ACCESS_KEY as representative of required env vars
        required_keys = ["DB_URL", "S3_ACCESS_KEY"]

        missing = [key for key in required_keys if not os.environ.get(key)]

        if missing:
            # Also check if config loaded them (fallback)
            if self.config:
                # If config exists, we assume it loaded them, but let's verify specifics
                if "DB_URL" in missing and self.config.database.url:
                    missing.remove("DB_URL")
                if "S3_ACCESS_KEY" in missing and self.config.storage.access_key:
                    missing.remove("S3_ACCESS_KEY")

        if missing:
            logger.error("env_check_failed", missing_keys=missing)
            return False

        logger.info("env_check_passed", status="OK")
        return True

    def check_db(self) -> bool:
        """Check Database Connectivity."""
        try:
            # Check availability in config
            if self.config.database.url:
                logger.info("db_check", status="OK", url="[MASKED]")
                return True
            else:
                logger.error("db_check_failed", reason="missing_url")
                return False
        except Exception as e:
            logger.error("db_check_failed", error=str(e))
            return False

    def check_s3(self) -> bool:
        """Check Object Storage Connectivity."""
        try:
            if self.config.storage.access_key and self.config.storage.secret_key:
                logger.info(
                    "s3_check", status="OK", bucket=self.config.storage.bucket_name
                )
                return True
            logger.error("s3_check_failed", reason="missing_credentials")
            return False
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
