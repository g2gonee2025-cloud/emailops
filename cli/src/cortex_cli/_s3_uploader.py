"""
S3/Spaces direct uploader module.

This module uploads files to DigitalOcean Spaces or any S3-compatible
storage using a thread pool and a shared boto3 client.

Key features:
- Concurrent uploads using a thread pool
- MIME type detection for Content-Type headers
- Error handling with per-file results
"""

import logging
import mimetypes
import threading
from collections.abc import Iterable, Iterator
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import boto3
from boto3.s3.transfer import TransferConfig
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
        self._client_lock = threading.Lock()
        self._transfer_config = TransferConfig(use_threads=False)

    def _get_s3_client(self) -> BaseClient:
        """
        Creates and returns an S3 client.
        """
        if self._s3_client is None:
            with self._client_lock:
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

    def _upload_file(self, local_path: Path, s3_key: str) -> tuple[str, int]:
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
            Config=self._transfer_config,
        )
        return s3_key, file_size

    @staticmethod
    def _normalize_prefix(prefix: str) -> str:
        if not prefix:
            return ""
        normalized = prefix.lstrip("/")
        if not normalized.endswith("/"):
            normalized = f"{normalized}/"
        return normalized

    def close(self) -> None:
        """Close the S3 client if possible."""
        if self._s3_client is None:
            return
        close_fn = getattr(self._s3_client, "close", None)
        if callable(close_fn):
            try:
                close_fn()
            except Exception:
                logger.warning("Failed to close S3 client", exc_info=True)
        self._s3_client = None

    def upload_files(
        self, source_dir: Path, files_to_upload: Iterable[Path], s3_prefix: str
    ) -> Iterator[tuple[bool, str]]:
        """
        Uploads a list of files to S3.

        Args:
            source_dir (Path): The root directory of the files.
            files_to_upload (Iterable[Path]): The list of file paths to upload.
            s3_prefix (str): The prefix to use for S3 keys.

        Yields:
            Iterator[Tuple[bool, str]]: A tuple of (success, message).
        """
        normalized_prefix = self._normalize_prefix(s3_prefix)
        files_iter = iter(files_to_upload)
        pending = {}

        try:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                while True:
                    while len(pending) < self.max_workers:
                        try:
                            local_path = next(files_iter)
                        except StopIteration:
                            break
                        try:
                            relative_path = local_path.relative_to(source_dir)
                        except ValueError:
                            logger.warning(
                                "Skipping file outside source directory: %s",
                                local_path,
                            )
                            yield False, f"{local_path}: not under source directory"
                            continue
                        s3_key = f"{normalized_prefix}{relative_path.as_posix()}"
                        future = executor.submit(self._upload_file, local_path, s3_key)
                        pending[future] = s3_key

                    if not pending:
                        break

                    done, _ = wait(pending, return_when=FIRST_COMPLETED)
                    for future in done:
                        s3_key = pending.pop(future)
                        try:
                            _, file_size = future.result()
                            yield True, f"{s3_key} ({file_size} bytes)"
                        except (ClientError, BotoCoreError) as exc:
                            logger.warning(
                                "Upload failed for %s", s3_key, exc_info=True
                            )
                            yield (
                                False,
                                (f"{s3_key}: upload failed ({exc.__class__.__name__})"),
                            )
                        except Exception:
                            logger.exception("Unexpected error uploading %s", s3_key)
                            yield False, f"{s3_key}: upload failed"
        finally:
            self.close()
