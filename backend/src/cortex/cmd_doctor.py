"""
Doctor Command Module.

Implements system health checks for the Cortex Doctor CLI.
"""

import os
import boto3
import structlog
from sqlalchemy import create_engine
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
            logger.error("env_check_failed", missing_keys=missing)
            return False

        logger.info("env_check_passed", status="OK")
        return True

    def check_db(self) -> bool:
        """Check Database Connectivity."""
        try:
            engine = create_engine(self.config.database.url)
            with engine.connect() as connection:
                logger.info("db_check", status="OK", url="[MASKED]")
                return True
        except Exception as e:
            logger.error("db_check_failed", error=str(e))
            return False

    def check_s3(self) -> bool:
        """Check Object Storage Connectivity."""
        try:
            s3 = boto3.client(
                "s3",
                aws_access_key_id=self.config.storage.access_key,
                aws_secret_access_key=self.config.storage.secret_key,
                endpoint_url=self.config.storage.endpoint_url,
                region_name=self.config.storage.region,
            )
            s3.list_buckets()
            logger.info("s3_check", status="OK", bucket=self.config.storage.bucket_name)
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
