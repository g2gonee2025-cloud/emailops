"""
S3/Spaces direct uploader module.

This module provides a robust and efficient way to upload files to
DigitalOcean Spaces or any S3-compatible storage. It is designed to be
used by the Cortex CLI, but can also be used as a standalone module.

Key features:
- Concurrent uploads using a thread pool
- Progress tracking with ETA calculation
- Graceful error handling and detailed summary reporting
- MIME type detection for proper Content-Type headers
- Configurable S3 client with retry mechanism

Classes:
    S3Uploader: A class that encapsulates the upload logic.

"""

import logging
import mimetypes
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterator, Tuple

import boto3
from botocore.client import BaseClient
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError

# Set up logging
logger = logging.getLogger(__name__)


class S3Uploader:
    """
    Handles uploading files to S3/Spaces with progress and error reporting.
    """

    def __init__(
        self,
        endpoint_url: str,
        region_name: str,
        access_key: str,
        secret_key: str,
        bucket_name: str,
        max_workers: int = 10,
    ):
        """
        Initializes the S3Uploader.

        Args:
            endpoint_url (str): The S3 endpoint URL.
            region_name (str): The S3 region name.
            access_key (str): The S3 access key.
            secret_key (str): The S3 secret key.
            bucket_name (str): The S3 bucket name.
            max_workers (int): The maximum number of concurrent upload workers.
        """
        self.endpoint_url = endpoint_url
        self.region_name = region_name
        self.access_key = access_key
        self.secret_key = secret_key
        self.bucket_name = bucket_name
        self.max_workers = max_workers
        self._s3_client = None

    def _get_s3_client(self) -> BaseClient:
        """
        Creates and returns an S3 client.
        """
        if self._s3_client is None:
            self._s3_client = boto3.client(
                "s3",
                endpoint_url=self.endpoint_url,
                region_name=self.region_name,
                aws_access_key_id=self.access_key,
                aws_secret_access_key=self.secret_key,
                config=Config(
                    signature_version="s3v4",
                    retries={"max_attempts": 3, "mode": "adaptive"},
                    connect_timeout=30,
                    read_timeout=60,
                ),
            )
        return self._s3_client

    def _get_content_type(self, file_path: Path) -> str:
        """
        Determines the content type of a file based on its extension.
        """
        content_type, _ = mimetypes.guess_type(str(file_path))
        return content_type or "application/octet-stream"

    def _upload_file(self, local_path: Path, s3_key: str) -> Tuple[str, int]:
        """
        Uploads a single file to S3.
        """
        s3_client = self._get_s3_client()
        content_type = self._get_content_type(local_path)
        file_size = local_path.stat().st_size
        s3_client.upload_file(
            str(local_path),
            self.bucket_name,
            s3_key,
            ExtraArgs={"ContentType": content_type},
        )
        return s3_key, file_size

    def upload_files(
        self, source_dir: Path, files_to_upload: list[Path], s3_prefix: str
    ) -> Iterator[Tuple[bool, str]]:
        """
        Uploads a list of files to S3.

        Args:
            source_dir (Path): The root directory of the files.
            files_to_upload (list[Path]): The list of file paths to upload.
            s3_prefix (str): The prefix to use for S3 keys.

        Yields:
            Iterator[Tuple[bool, str]]: A tuple of (success, message).
        """
        s3_tasks = [
            (p, f"{s3_prefix}{p.relative_to(source_dir).as_posix()}")
            for p in files_to_upload
        ]

        if not s3_tasks:
            return

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._upload_file, local_path, s3_key): s3_key
                for local_path, s3_key in s3_tasks
            }

            for future in as_completed(futures):
                s3_key = futures[future]
                try:
                    _, file_size = future.result()
                    yield True, f"{s3_key} ({file_size} bytes)"
                except (ClientError, BotoCoreError) as e:
                    yield False, f"{s3_key}: {e}"
                except Exception as e:
                    yield False, f"{s3_key}: An unexpected error occurred: {e}"
