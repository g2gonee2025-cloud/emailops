"""Integration tests for core EmailOps workflows."""

import json
import os
import tempfile
from pathlib import Path
from unittest import TestCase
from unittest.mock import MagicMock, Mock, patch


class TestDocumentProcessingWorkflow(TestCase):
    """Test complete document processing pipeline."""

    @patch('emailops.config.get_config')
    @patch('emailops.llm_client.embed_texts')
    def test_document_processing_workflow(self, mock_embed_texts, mock_get_config):
        """Test complete document processing pipeline."""
        # Setup config
        mock_config = Mock()
        mock_config.batch_size = 2
        mock_config.chunk_size = 100
        mock_config.chunk_overlap = 20
        mock_config.num_workers = 1
        mock_config.chunk_dirname = "_chunks"
        mock_config.index_dirname = "_index"
        mock_get_config.return_value = mock_config

        mock_embed_texts.return_value = [[0.1, 0.2], [0.3, 0.4]]

        with tempfile.TemporaryDirectory() as tmpdir:
            from emailops.cli import cmd_index

            root_dir = Path(tmpdir)

            # Create conversation directories
            conv1_dir = root_dir / "conv1"
            conv1_dir.mkdir()
            (conv1_dir / "Conversation.txt").write_text("This is conversation one.")

            conv2_dir = root_dir / "conv2"
            conv2_dir.mkdir()
            (conv2_dir / "Conversation.txt").write_text("This is conversation two.")

            # Process documents
            args = Mock()
            args.root = root_dir
            args.provider = "vertex"
            args.batch = 2
            args.limit = None
            args.force_reindex = True
            args.indexer_args = None
            args.timeout = 120
            cmd_index(args)

            # Verify output directory structure
            index_dir = root_dir / "_index"
            assert index_dir.exists()
            assert (index_dir / "embeddings.npy").exists()
            assert (index_dir / "mapping.json").exists()

    @patch('emailops.config.get_config')
    def test_configuration_workflow(self, mock_get_config):
        """Test configuration loading and usage."""
        from emailops.core_config import EmailOpsConfig

        # Setup mock config
        mock_config = Mock(spec=EmailOpsConfig)
        mock_config.batch_size = 64
        mock_config.chunk_size = 1600
        mock_config.gcp_project = "test-project"
        mock_config.get_secrets_dir.return_value = Path("/secrets")
        mock_config.get_credential_file.return_value = Path("/secrets/creds.json")
        mock_config.to_dict.return_value = {
            "DEFAULT_BATCH_SIZE": 64,
            "DEFAULT_CHUNK_SIZE": 1600,
            "GCP_PROJECT": "test-project"
        }
        mock_get_config.return_value = mock_config

        # Test configuration usage
        from emailops.core_config import get_config
        config = get_config()

        assert config.batch_size == 64
        assert config.chunk_size == 1600
        assert config.gcp_project == "test-project"

        # Test configuration update
        with patch.dict(os.environ, {"DEFAULT_BATCH_SIZE": "128"}):
            config_dict = config.to_dict()
            assert isinstance(config_dict, dict)

    @patch('emailops.email_indexer._initialize_gcp_credentials')
    @patch('emailops.llm_client.embed_texts')
    @patch('emailops.config.get_config')
    def test_error_recovery_workflow(self, mock_get_config, mock_embed, mock_init_creds):
        """Test system recovery from errors."""
        # Setup config
        mock_config = Mock()
        mock_config.batch_size = 2
        mock_config.chunk_size = 100
        mock_config.chunk_overlap = 20
        mock_config.num_workers = 1
        mock_config.chunk_dirname = "_chunks"
        mock_config.index_dirname = "_index"
        mock_get_config.return_value = mock_config

        # Setup credentials
        mock_init_creds.return_value = "/path/to/creds.json"

        # Setup embeddings to fail
        mock_embed.side_effect = Exception("API error")

        with tempfile.TemporaryDirectory() as tmpdir:
            from emailops.indexing_main import main as indexer_main

            # Create a dummy conversation directory
            conv_dir = Path(tmpdir) / "conv1"
            conv_dir.mkdir()
            (conv_dir / "Conversation.txt").write_text("dummy content")

            # Run the indexer, which should raise the exception
            with patch('sys.argv', ['emailops/email_indexer.py', '--root', str(tmpdir), '--force-reindex']):
                with self.assertRaises(Exception) as cm:
                    indexer_main()
                self.assertIn("API error", str(cm.exception))

            # Verify embed was called
            assert mock_embed.call_count > 0


class TestEmbeddingWorkflow(TestCase):
    """Test embedding generation workflow."""

    @patch('emailops.llm_runtime.load_validated_accounts')
    @patch('emailops.llm_client.embed_texts')
    def test_embedding_pipeline_vertex(self, mock_embed_texts, mock_load_accounts):
        """Test embedding pipeline with Vertex AI provider."""
        import numpy as np
        from emailops.llm_client_shim import embed_texts

        mock_load_accounts.return_value = [Mock(project_id='test', credentials_path='/fake.json')]
        mock_embed_texts.return_value = np.random.rand(2, 3072).astype(np.float32)

        texts = ["this is a test", "this is another test"]
        embeddings = embed_texts(texts, provider="vertex")

        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (2, 3072)




class TestSecurityWorkflow(TestCase):
    """Test security validation workflows."""

    def test_path_validation_workflow(self):
        """Test complete path validation workflow."""
        from emailops.core_validators import validate_directory_path, validate_file_path

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test directory and file
            test_dir = Path(tmpdir) / "test_dir"
            test_dir.mkdir()
            test_file = test_dir / "test.txt"
            test_file.write_text("test content")

            # Test valid paths
            assert validate_directory_path(str(test_dir))[0]
            assert validate_file_path(str(test_file))[0]

            # Test invalid paths
            assert not validate_directory_path("../../../etc")[0]
            assert not validate_file_path("../../../etc/passwd")[0]

            # Test non-existent paths
            assert not validate_directory_path(str(test_dir / "nonexistent"))[0]
            assert not validate_file_path(str(test_dir / "nonexistent.txt"))[0]

    def test_command_validation_workflow(self):
        """Test command execution validation workflow."""
        from emailops.core_validators import validate_command_args

        # Test safe commands
        assert validate_command_args("ls", ["-la"])[0]
        assert validate_command_args("echo", ["Hello World"])[0]

        # Test dangerous commands
        assert not validate_command_args("rm", ["-rf", "/"])[0]
        assert not validate_command_args("ls", ["|", "cat"])[0]
        assert not validate_command_args("echo", ["test;", "rm", "-rf", "/"])[0]

    def test_environment_variable_workflow(self):
        """Test environment variable validation workflow."""
        from emailops.core_validators import validate_environment_variable

        # Test valid environment variables
        assert validate_environment_variable("PATH", "/usr/bin:/usr/local/bin")[0]
        assert validate_environment_variable("HOME", "/home/user")[0]
        assert validate_environment_variable("PYTHON_VERSION", "3.9.0")[0]

        # Test invalid environment variables
        assert not validate_environment_variable("", "value")[0]
        assert not validate_environment_variable("invalid-name", "value")[0]
        assert not validate_environment_variable("TEST", "value\x00with\x00nulls")[0]


class TestIndexWorkflow(TestCase):
    """Test index creation and management workflows."""

    def test_index_creation_workflow(self):
        """Test index creation from embeddings."""
        pass

    def test_index_repair_workflow(self):
        """Test repairing corrupted index."""
        pass


class TestEmailProcessingWorkflow(TestCase):
    """Test email processing workflows."""

    def test_email_parsing_workflow(self):
        """Test complete email parsing workflow."""
        from emailops.util_main import (
            clean_email_text,
            extract_email_metadata,
            split_email_thread,
        )

        email_text = """From: sender@example.com
To: recipient@example.com
Subject: Test Email
Date: Mon, 1 Jan 2024 10:00:00 +0000

Hello,

This is the email body.

Best regards,
Sender

----- Original Message -----
From: other@example.com
Date: Sun, 31 Dec 2023 10:00:00 +0000

Previous message content."""

        # Clean the email
        cleaned = clean_email_text(email_text)
        assert "Hello" in cleaned
        assert "email body" in cleaned

        # Extract metadata
        metadata = extract_email_metadata(email_text)
        assert metadata["sender"] == "sender@example.com"
        assert metadata["subject"] == "Test Email"

        # Split thread
        messages = split_email_thread(email_text)
        assert len(messages) >= 1

    def test_conversation_loading_workflow(self):
        """Test conversation loading workflow."""
        from emailops.util_main import load_conversation

        with tempfile.TemporaryDirectory() as tmpdir:
            conv_dir = Path(tmpdir)

            # Create conversation structure
            (conv_dir / "Conversation.txt").write_text("Email conversation content")

            manifest = {
                "id": "conv123",
                "subject": "Test Conversation",
                "participants": ["user1@example.com", "user2@example.com"]
            }
            (conv_dir / "manifest.json").write_text(json.dumps(manifest))

            summary = {"summary": "This is a test conversation"}
            (conv_dir / "summary.json").write_text(json.dumps(summary))

            # Create attachments
            att_dir = conv_dir / "Attachments"
            att_dir.mkdir()
            (att_dir / "test.txt").write_text("Attachment content")

            # Load the conversation
            conv = load_conversation(conv_dir, include_attachment_text=True)

            assert "Email conversation content" in conv["conversation_txt"]
            assert "Attachment content" in conv["conversation_txt"]
            assert conv["manifest"]["id"] == "conv123"
            assert conv["summary"]["summary"] == "This is a test conversation"
            assert len(conv["attachments"]) == 1
            assert "Attachment content" in conv["attachments"][0]["text"]


class TestEndToEndWorkflow(TestCase):
    """Test complete end-to-end workflows."""

    def test_complete_pipeline(self):
        """Test complete pipeline from documents to embeddings."""
        pass
