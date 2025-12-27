"""
S3/Spaces Source Handler for Ingestion.

Implements §6 and §17 of the Canonical Blueprint.
Downloads conversation folders from DigitalOcean Spaces for processing.
"""

from __future__ import annotations

import logging
import os
import tempfile
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from cortex.config.loader import get_config
from cortex.ingestion.core_manifest import parse_manifest_text
from cortex.ingestion.text_utils import strip_control_chars
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class S3ConversationFolder(BaseModel):
    """Represents a conversation folder in S3."""

    prefix: str  # e.g., "outlook/EML-2024-12-01_ABC123 - Subject/"
    name: str  # e.g., "EML-2024-12-01_ABC123 - Subject"
    files: list[str] = Field(default_factory=list)  # List of file keys in this folder
    last_modified: datetime | None = None  # Max mtime of files in folder


class S3SourceHandler:
    """
    Handler for reading conversation data from S3/Spaces.

    Blueprint §6.1: source_type="s3"
    Blueprint §17.3: DigitalOcean Spaces (S3-compatible)
    """

    def __init__(
        self,
        endpoint_url: str | None = None,
        region: str | None = None,
        access_key: str | None = None,
        secret_key: str | None = None,
        bucket: str | None = None,
    ):
        """Initialize S3 client with config or explicit params."""
        config = get_config()

        self.endpoint_url = endpoint_url or config.storage.endpoint_url
        self.region = region or config.storage.region
        self.bucket = bucket or config.storage.bucket_raw

        # Store credentials with a leading underscore to mark them as "internal"
        # and reduce the risk of accidental leakage in logs.
        self._access_key = access_key or config.storage.access_key
        self._secret_key = secret_key or config.storage.secret_key

        self._client: Any | None = None

    @property
    def client(self) -> Any:
        """Lazy-load S3 client."""
        if self._client is None:
            self._client = boto3.client(
                "s3",
                endpoint_url=self.endpoint_url,
                region_name=self.region,
                aws_access_key_id=self._access_key,
                aws_secret_access_key=self._secret_key,
                config=Config(
                    signature_version="s3v4",
                    retries={"max_attempts": 3, "mode": "adaptive"},
                ),
            )
        return self._client

    def close(self) -> None:
        """Close the S3 client."""
        if self._client:
            try:
                self._client.close()
            except Exception:
                pass
            self._client = None

    def __enter__(self) -> S3SourceHandler:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def list_conversation_folders(
        self,
        prefix: str = "Outlook/",
        limit: int | None = None,
    ) -> Iterator[S3ConversationFolder]:
        """
        Efficiently list conversation folders under a prefix using a single pass.

        This method avoids the N+1 API call pattern by listing all objects
        and grouping them by folder prefix in a streaming manner.

        Args:
            prefix: S3 prefix to search under (default: raw/outlook/)
            limit: Maximum number of folders to return

        Yields:
            S3ConversationFolder objects
        """
        logger.info(
            f"Efficiently listing conversation folders in s3://{self.bucket}/{prefix}"
        )

        paginator = self.client.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=self.bucket, Prefix=prefix)

        current_folder_prefix: str | None = None
        current_files: list[str] = []
        current_max_mtime: datetime | None = None
        folder_count = 0

        for page in pages:
            for obj in page.get("Contents", []):
                key = obj["Key"]

                # Determine the folder prefix for the object
                if "/" not in key:
                    continue  # Skip objects at the root of the prefix
                folder_prefix = key.rsplit("/", 1)[0] + "/"

                if folder_prefix != current_folder_prefix:
                    # New folder detected, yield the completed previous one
                    if current_folder_prefix:
                        folder_name = current_folder_prefix.rstrip("/").split("/")[-1]
                        yield S3ConversationFolder(
                            prefix=current_folder_prefix,
                            name=folder_name,
                            files=current_files,
                            last_modified=current_max_mtime,
                        )
                        folder_count += 1
                        if limit and folder_count >= limit:
                            return

                    # Start tracking the new folder
                    current_folder_prefix = folder_prefix
                    current_files = []
                    current_max_mtime = None

                # Add file and update last_modified for the current folder
                current_files.append(key)
                last_modified = obj.get("LastModified")
                if last_modified and (
                    current_max_mtime is None or last_modified > current_max_mtime
                ):
                    current_max_mtime = last_modified

        # Yield the last folder after the loop
        if current_folder_prefix and (not limit or folder_count < limit):
            folder_name = current_folder_prefix.rstrip("/").split("/")[-1]
            yield S3ConversationFolder(
                prefix=current_folder_prefix,
                name=folder_name,
                files=current_files,
                last_modified=current_max_mtime,
            )

    def build_folder(self, prefix: str) -> S3ConversationFolder:
        """
        Build an S3ConversationFolder for an explicit prefix.

        This is used by ingestion jobs when a specific folder prefix is
        provided via IngestJob.source_uri.
        """
        normalized_prefix = prefix if prefix.endswith("/") else f"{prefix}/"
        name = normalized_prefix.rstrip("/").split("/")[-1]

        files = []
        max_mtime = None
        paginator = self.client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=normalized_prefix):
            for obj in page.get("Contents", []):
                files.append(obj["Key"])
                if "LastModified" in obj:
                    mt = obj["LastModified"]
                    if max_mtime is None or mt > max_mtime:
                        max_mtime = mt

        return S3ConversationFolder(
            prefix=normalized_prefix, name=name, files=files, last_modified=max_mtime
        )

    def download_conversation_folder(
        self,
        folder: S3ConversationFolder,
        local_dir: Path | None = None,
    ) -> Path:
        """
        Download a conversation folder to local disk.

        Args:
            folder: The S3ConversationFolder to download
            local_dir: Optional local directory (creates temp if not specified)

        Returns:
            Path to the downloaded folder
        """
        if local_dir is None:
            local_dir = Path(tempfile.mkdtemp(prefix="cortex_ingest_"))

        folder_path = local_dir / folder.name
        folder_path.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"Downloading {len(folder.files)} files from {folder.prefix} to {folder_path}"
        )

        for file_key in folder.files:
            if not file_key.startswith(folder.prefix):
                logger.warning(
                    f"Skipping file '{file_key}' outside of folder '{folder.prefix}'"
                )
                continue

            relative_path_str = file_key[len(folder.prefix) :]
            if not relative_path_str:
                continue

            # Construct the full local path and resolve any ".." components.
            local_file = (folder_path / relative_path_str).resolve()

            # Robust path traversal check: Ensure the resolved file path is
            # a child of the resolved folder path.
            if folder_path.resolve() not in local_file.parents:
                logger.error(
                    "Path traversal attempt detected. "
                    f"File key '{file_key}' resolves outside of target directory '{folder_path.resolve()}'"
                )
                continue

            local_file.parent.mkdir(parents=True, exist_ok=True)

            try:
                self.client.download_file(self.bucket, file_key, str(local_file))
            except ClientError as e:
                logger.error(f"Failed to download s3://{self.bucket}/{file_key}: {e}")
                # Re-raise the exception to halt processing of an incomplete folder
                raise

        return folder_path

    def get_object_content(self, key: str) -> bytes:
        """Get raw content of an S3 object."""
        response = self.client.get_object(Bucket=self.bucket, Key=key)
        return response["Body"].read()

    def get_json_object(self, key: str) -> dict[str, Any]:
        """Get and parse a JSON object from S3."""
        content = self.get_object_content(key)
        decoded = strip_control_chars(content.decode("utf-8-sig"))
        return parse_manifest_text(decoded, source=key)

    def get_text_object(self, key: str) -> str:
        """Get text content of an S3 object."""
        content = self.get_object_content(key)
        return content.decode("utf-8-sig")

    def upload_file(self, local_path: Path, key: str) -> None:
        """Upload a local file to S3.

        Args:
            local_path: Path to local file to upload
            key: S3 key (path) for the uploaded file
        """
        self.client.upload_file(str(local_path), self.bucket, key)

    def conversation_exists(self, folder_prefix: str) -> bool:
        """Check if a conversation folder exists and has required files."""
        try:
            # Check for conversation.txt
            conv_key = f"{folder_prefix}conversation.txt"
            self.client.head_object(Bucket=self.bucket, Key=conv_key)
            return True
        except Exception:
            return False

    def list_objects(self, prefix: str) -> Iterator[dict[str, Any]]:
        """
        List all objects under a given prefix.
        Args:
            prefix: S3 prefix to search under.
        Yields:
            Object dictionaries from boto3.
        """
        paginator = self.client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for content in page.get("Contents", []):
                yield content


def create_s3_source() -> S3SourceHandler:
    """Factory function to create S3 source handler from config."""
    return S3SourceHandler()
