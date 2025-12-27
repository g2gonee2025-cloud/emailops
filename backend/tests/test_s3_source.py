import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from cortex.ingestion.s3_source import S3ConversationFolder, S3SourceHandler


class TestS3SourceHandler(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.handler = S3SourceHandler(
            endpoint_url="http://localhost:9000",
            bucket="test-bucket",
            access_key="test",
            secret_key="test",
        )
        self.handler._client = MagicMock()

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_download_folder_path_traversal(self):
        # Malicious file key attempting path traversal
        malicious_file_key = "raw/folder/../../../etc/passwd"
        safe_file_key = "raw/folder/safe_file.txt"

        folder = S3ConversationFolder(
            prefix="raw/folder/",
            name="test_folder",
            files=[safe_file_key, malicious_file_key],
        )

        download_path = self.handler.download_conversation_folder(
            folder, local_dir=Path(self.temp_dir)
        )

        # Verify that the safe file was downloaded
        self.handler.client.download_file.assert_any_call(
            "test-bucket",
            safe_file_key,
            str(download_path / "safe_file.txt"),
        )
        # Verify that the malicious file was NOT downloaded
        malicious_download_path = str(Path(self.temp_dir) / "etc/passwd")
        for call in self.handler.client.download_file.call_args_list:
            self.assertNotEqual(call[0][2], malicious_download_path)


if __name__ == "__main__":
    unittest.main()
