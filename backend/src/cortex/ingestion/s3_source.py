"""
S3/Spaces Source Handler for Ingestion.

Implements §6 and §17 of the Canonical Blueprint.
Downloads conversation folders from DigitalOcean Spaces for processing.
"""
from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import boto3
from botocore.config import Config
from cortex.config.loader import get_config
from cortex.ingestion.core_manifest import parse_manifest_text
from cortex.ingestion.text_utils import strip_control_chars
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class S3ConversationFolder(BaseModel):
    """Represents a conversation folder in S3."""

    prefix: str  # e.g., "outlook/EML-2024-12-01_ABC123 - Subject/"
    name: str  # e.g., "EML-2024-12-01_ABC123 - Subject"
    files: List[str] = Field(default_factory=list)  # List of file keys in this folder


class S3SourceHandler:
    """
    Handler for reading conversation data from S3/Spaces.

    Blueprint §6.1: source_type="s3"
    Blueprint §17.3: DigitalOcean Spaces (S3-compatible)
    """

    def __init__(
        self,
        endpoint_url: Optional[str] = None,
        region: Optional[str] = None,
        access_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        bucket: Optional[str] = None,
    ):
        """Initialize S3 client with config or explicit params."""
        config = get_config()

        self.endpoint_url = endpoint_url or config.s3.endpoint_url
        self.region = region or config.s3.region
        self.access_key = access_key or config.s3.access_key
        self.secret_key = secret_key or config.s3.secret_key
        self.bucket = bucket or config.s3.bucket_raw

        self._client: Optional[Any] = None

    @property
    def client(self) -> Any:
        """Lazy-load S3 client."""
        if self._client is None:
            self._client = boto3.client(
                "s3",
                endpoint_url=self.endpoint_url,
                region_name=self.region,
                aws_access_key_id=self.access_key,
                aws_secret_access_key=self.secret_key,
                config=Config(
                    signature_version="s3v4",
                    retries={"max_attempts": 3, "mode": "adaptive"},
                ),
            )
        return self._client

    def list_conversation_folders(
        self,
        prefix: str = "Outlook/",
        limit: Optional[int] = None,
    ) -> Iterator[S3ConversationFolder]:
        """
        List conversation folders under a prefix.

        Each conversation folder is expected to contain:
        - conversation.txt (the email thread transcript)
        - manifest.json (metadata)
        - summary.json (optional, AI-generated summary)
        - attachments/ (optional subdirectory)

        Args:
            prefix: S3 prefix to search under (default: raw/outlook/)
            limit: Maximum number of folders to return

        Yields:
            S3ConversationFolder objects
        """
        logger.info(f"Listing conversation folders in s3://{self.bucket}/{prefix}")

        paginator = self.client.get_paginator("list_objects_v2")

        # Use delimiter to get "directories"
        seen_folders = set()
        folder_count = 0

        for page in paginator.paginate(
            Bucket=self.bucket, Prefix=prefix, Delimiter="/"
        ):
            # CommonPrefixes contains the "directories"
            for common_prefix in page.get("CommonPrefixes", []):
                folder_prefix = common_prefix["Prefix"]
                folder_name = folder_prefix.rstrip("/").split("/")[-1]

                if folder_prefix in seen_folders:
                    continue
                seen_folders.add(folder_prefix)

                # List files in this folder
                files = self._list_folder_files(folder_prefix)

                yield S3ConversationFolder(
                    prefix=folder_prefix,
                    name=folder_name,
                    files=files,
                )

                folder_count += 1
                if limit and folder_count >= limit:
                    return

    def _list_folder_files(self, prefix: str) -> List[str]:
        """List all files in a folder (recursively)."""
        files = []
        paginator = self.client.get_paginator("list_objects_v2")

        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                files.append(obj["Key"])

        return files

    def build_folder(self, prefix: str) -> S3ConversationFolder:
        """
        Build an S3ConversationFolder for an explicit prefix.

        This is used by ingestion jobs when a specific folder prefix is
        provided via IngestJob.source_uri.
        """
        normalized_prefix = prefix if prefix.endswith("/") else f"{prefix}/"
        name = normalized_prefix.rstrip("/").split("/")[-1]
        files = self._list_folder_files(normalized_prefix)
        return S3ConversationFolder(prefix=normalized_prefix, name=name, files=files)

    def download_conversation_folder(
        self,
        folder: S3ConversationFolder,
        local_dir: Optional[Path] = None,
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
            # Get relative path within folder
            relative_path = file_key[len(folder.prefix) :]
            if not relative_path:
                continue

            local_file = folder_path / relative_path
            local_file.parent.mkdir(parents=True, exist_ok=True)

            self.client.download_file(self.bucket, file_key, str(local_file))

        return folder_path

    def get_object_content(self, key: str) -> bytes:
        """Get raw content of an S3 object."""
        response = self.client.get_object(Bucket=self.bucket, Key=key)
        return response["Body"].read()

    def get_json_object(self, key: str) -> Dict[str, Any]:
        """Get and parse a JSON object from S3."""
        content = self.get_object_content(key)
        decoded = strip_control_chars(content.decode("utf-8-sig"))
        return parse_manifest_text(decoded, source=key)

    def get_text_object(self, key: str) -> str:
        """Get text content of an S3 object."""
        content = self.get_object_content(key)
        return content.decode("utf-8-sig")

    def conversation_exists(self, folder_prefix: str) -> bool:
        """Check if a conversation folder exists and has required files."""
        try:
            # Check for conversation.txt
            conv_key = f"{folder_prefix}conversation.txt"
            self.client.head_object(Bucket=self.bucket, Key=conv_key)
            return True
        except Exception:
            return False


def create_s3_source() -> S3SourceHandler:
    """Factory function to create S3 source handler from config."""
    return S3SourceHandler()
