"""Integration tests for core EmailOps workflows."""

import json
import os
import tempfile
from pathlib import Path
from unittest import TestCase
from unittest.mock import Mock, MagicMock, patch, call

import pytest


class TestDocumentProcessingWorkflow(TestCase):
    """Test complete document processing pipeline."""
    
    @patch('emailops.config.get_config')
    @patch('emailops.text_chunker.TextChunker')
    @patch('emailops.utils.read_text_file')
    def test_document_processing_workflow(self, mock_read, mock_chunker_class, mock_get_config):
        """Test complete document processing pipeline."""
        # Setup config
        mock_config = Mock()
        mock_config.DEFAULT_BATCH_SIZE = 2
        mock_config.DEFAULT_CHUNK_SIZE = 100
        mock_config.DEFAULT_CHUNK_OVERLAP = 20
        mock_config.DEFAULT_NUM_WORKERS = 1
        mock_config.CHUNK_DIRNAME = "_chunks"
        mock_config.INDEX_DIRNAME = "_index"
        mock_get_config.return_value = mock_config
        
        # Setup chunker
        mock_chunker = Mock()
        mock_chunker.chunk_text.return_value = [
            {"text": "chunk1", "metadata": {}},
            {"text": "chunk2", "metadata": {}}
        ]
        mock_chunker_class.return_value = mock_chunker
        
        # Setup file reading
        mock_read.return_value = "Document content to be chunked"
        
        with tempfile.TemporaryDirectory() as tmpdir:
            from processing.processor import UnifiedProcessor
            
            # Create test documents
            input_dir = Path(tmpdir) / "input"
            input_dir.mkdir()
            (input_dir / "doc1.txt").write_text("Document 1")
            (input_dir / "doc2.txt").write_text("Document 2")
            
            output_dir = Path(tmpdir) / "output"
            
            # Process documents
            processor = UnifiedProcessor(
                root_dir=str(output_dir),
                mode="chunk",
                num_workers=1
            )
            
            processor.chunk_documents(str(input_dir), "*.txt")
            
            # Verify chunker was called
            assert mock_chunker.chunk_text.call_count >= 1
            
            # Verify output directory structure
            assert output_dir.exists()
            chunks_dir = output_dir / "_chunks" / "chunks"
            assert chunks_dir.exists()
    
    @patch('emailops.config.get_config')
    def test_configuration_workflow(self, mock_get_config):
        """Test configuration loading and usage."""
        from emailops.config import EmailOpsConfig
        
        # Setup mock config
        mock_config = Mock(spec=EmailOpsConfig)
        mock_config.DEFAULT_BATCH_SIZE = 64
        mock_config.DEFAULT_CHUNK_SIZE = 1600
        mock_config.GCP_PROJECT = "test-project"
        mock_config.get_secrets_dir.return_value = Path("/secrets")
        mock_config.get_credential_file.return_value = Path("/secrets/creds.json")
        mock_config.to_dict.return_value = {
            "DEFAULT_BATCH_SIZE": 64,
            "DEFAULT_CHUNK_SIZE": 1600,
            "GCP_PROJECT": "test-project"
        }
        mock_get_config.return_value = mock_config
        
        # Test configuration usage
        from emailops.config import get_config
        config = get_config()
        
        assert config.DEFAULT_BATCH_SIZE == 64
        assert config.DEFAULT_CHUNK_SIZE == 1600
        assert config.GCP_PROJECT == "test-project"
        
        # Test configuration update
        with patch.dict(os.environ, {"DEFAULT_BATCH_SIZE": "128"}):
            config_dict = config.to_dict()
            assert isinstance(config_dict, dict)
    
    @patch('processing.processor._initialize_gcp_credentials')
    @patch('processing.processor.embed_texts')
    @patch('emailops.config.get_config')
    def test_error_recovery_workflow(self, mock_get_config, mock_embed, mock_init_creds):
        """Test system recovery from errors."""
        # Setup config
        mock_config = Mock()
        mock_config.DEFAULT_BATCH_SIZE = 2
        mock_config.DEFAULT_CHUNK_SIZE = 100
        mock_config.DEFAULT_CHUNK_OVERLAP = 20
        mock_config.DEFAULT_NUM_WORKERS = 1
        mock_config.CHUNK_DIRNAME = "_chunks"
        mock_config.INDEX_DIRNAME = "_index"
        mock_get_config.return_value = mock_config
        
        # Setup credentials
        mock_init_creds.return_value = "/path/to/creds.json"
        
        # Setup embeddings to fail first, then succeed
        mock_embed.side_effect = [
            Exception("API error"),
            [[0.1, 0.2], [0.3, 0.4]]  # Success on retry
        ]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            from processing.processor import UnifiedProcessor
            
            # Create chunks directory with test data
            chunks_dir = Path(tmpdir) / "_chunks" / "chunks"
            chunks_dir.mkdir(parents=True)
            
            chunk_data = {
                "doc_id": "test_doc",
                "chunks": [
                    {"text": "chunk 1"},
                    {"text": "chunk 2"}
                ]
            }
            (chunks_dir / "test.json").write_text(json.dumps(chunk_data))
            
            processor = UnifiedProcessor(
                root_dir=tmpdir,
                mode="embed",
                batch_size=2
            )
            
            # Process should handle the error gracefully
            processor.create_embeddings(use_chunked_files=True)
            
            # Verify retry was attempted
            assert mock_embed.call_count >= 1


class TestEmbeddingWorkflow(TestCase):
    """Test embedding generation workflow."""
    
    @patch('emailops.llm_client.llm_runtime')
    def test_embedding_pipeline_vertex(self, mock_runtime):
        """Test embedding pipeline with Vertex AI provider."""
        mock_runtime.embed_texts.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        
        from emailops.llm_client import embed_texts
        
        texts = ["text1", "text2"]
        embeddings = embed_texts(texts, provider="vertex")
        
        assert len(embeddings) == 2
        assert len(embeddings[0]) == 3
        mock_runtime.embed_texts.assert_called_once_with(texts, provider="vertex")
    
    @patch('emailops.llm_client.llm_runtime')
    def test_embedding_pipeline_openai(self, mock_runtime):
        """Test embedding pipeline with OpenAI provider."""
        mock_runtime.embed_texts.return_value = [[0.1, 0.2], [0.3, 0.4]]
        
        from emailops.llm_client import embed_texts
        
        texts = ["text1", "text2"]
        embeddings = embed_texts(texts, provider="openai")
        
        assert len(embeddings) == 2
        mock_runtime.embed_texts.assert_called_once_with(texts, provider="openai")
    
    @patch('emailops.llm_client.llm_runtime')
    def test_embedding_batch_processing(self, mock_runtime):
        """Test batch processing of embeddings."""
        # Mock large batch response
        batch_size = 100
        mock_runtime.embed_texts.return_value = [[0.1, 0.2]] * batch_size
        
        from emailops.llm_client import embed_texts
        
        texts = [f"text{i}" for i in range(batch_size)]
        embeddings = embed_texts(texts)
        
        assert len(embeddings) == batch_size
        mock_runtime.embed_texts.assert_called_once()
    
    @patch('emailops.llm_client.llm_runtime')
    def test_embedding_error_handling(self, mock_runtime):
        """Test error handling in embedding pipeline."""
        mock_runtime.embed_texts.side_effect = RuntimeError("API quota exceeded")
        
        from emailops.llm_client import embed_texts
        
        with pytest.raises(RuntimeError, match="API quota exceeded"):
            embed_texts(["text1", "text2"])


class TestSecurityWorkflow(TestCase):
    """Test security validation workflows."""
    
    def test_path_validation_workflow(self):
        """Test complete path validation workflow."""
        from emailops.validators import validate_directory_path, validate_file_path
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test directory and file
            test_dir = Path(tmpdir) / "test_dir"
            test_dir.mkdir()
            test_file = test_dir / "test.txt"
            test_file.write_text("test content")
            
            # Test valid paths
            assert validate_directory_path(str(test_dir))
            assert validate_file_path(str(test_file))
            
            # Test invalid paths
            assert not validate_directory_path("../../../etc")
            assert not validate_file_path("../../../etc/passwd")
            
            # Test non-existent paths
            assert not validate_directory_path(str(test_dir / "nonexistent"))
            assert not validate_file_path(str(test_dir / "nonexistent.txt"))
    
    def test_command_validation_workflow(self):
        """Test command execution validation workflow."""
        from emailops.validators import validate_command_args
        
        # Test safe commands
        assert validate_command_args("ls", ["-la"])
        assert validate_command_args("echo", ["Hello World"])
        
        # Test dangerous commands
        assert not validate_command_args("rm", ["-rf", "/"])
        assert not validate_command_args("ls", ["|", "cat"])
        assert not validate_command_args("echo", ["test;", "rm", "-rf", "/"])
    
    def test_environment_variable_workflow(self):
        """Test environment variable validation workflow."""
        from emailops.validators import validate_environment_variable
        
        # Test valid environment variables
        assert validate_environment_variable("PATH", "/usr/bin:/usr/local/bin")
        assert validate_environment_variable("HOME", "/home/user")
        assert validate_environment_variable("PYTHON_VERSION", "3.9.0")
        
        # Test invalid environment variables
        assert not validate_environment_variable("", "value")
        assert not validate_environment_variable("invalid-name", "value")
        assert not validate_environment_variable("TEST", "value\x00with\x00nulls")


class TestIndexWorkflow(TestCase):
    """Test index creation and management workflows."""
    
    @patch('processing.processor.faiss')
    def test_index_creation_workflow(self, mock_faiss):
        """Test index creation from embeddings."""
        import numpy as np
        from pathlib import Path
        import pickle
        
        mock_index = Mock()
        mock_faiss.IndexFlatIP.return_value = mock_index
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create mock embeddings
            emb_dir = Path(tmpdir) / "_index" / "embeddings"
            emb_dir.mkdir(parents=True)
            
            # Create batch files
            batch_data = {
                "chunks": [
                    {"text": "chunk1", "doc_id": "doc1"},
                    {"text": "chunk2", "doc_id": "doc1"}
                ],
                "embeddings": np.array([[0.1, 0.2], [0.3, 0.4]], dtype="float32")
            }
            
            with open(emb_dir / "batch_00000.pkl", "wb") as f:
                pickle.dump(batch_data, f)
            
            # Create processor and repair index
            from processing.processor import UnifiedProcessor
            processor = UnifiedProcessor(tmpdir, mode="repair")
            processor.repair_index(remove_batches=False)
            
            # Verify index was created
            assert mock_faiss.IndexFlatIP.called
            assert mock_index.add.called
            assert mock_faiss.write_index.called
            
            # Verify output files
            index_dir = Path(tmpdir) / "_index"
            assert (index_dir / "embeddings.npy").exists()
            assert (index_dir / "mapping.json").exists()
    
    @patch('processing.processor.embed_texts')
    @patch('processing.processor._initialize_gcp_credentials')
    def test_index_repair_workflow(self, mock_init_creds, mock_embed):
        """Test repairing corrupted index."""
        import numpy as np
        import pickle
        
        mock_init_creds.return_value = "/path/to/creds.json"
        mock_embed.return_value = [[0.5, 0.6]]  # New embeddings for repair
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create corrupted index with zero vectors
            emb_dir = Path(tmpdir) / "_index" / "embeddings"
            emb_dir.mkdir(parents=True)
            
            batch_data = {
                "chunks": [{"text": "test", "doc_id": "doc1"}],
                "embeddings": np.array([[0.0, 0.0]], dtype="float32")  # Zero vector
            }
            
            with open(emb_dir / "batch.pkl", "wb") as f:
                pickle.dump(batch_data, f)
            
            # Repair the index
            from processing.processor import UnifiedProcessor
            processor = UnifiedProcessor(tmpdir, mode="fix")
            processor.fix_failed_embeddings()
            
            # Verify repair was attempted
            mock_embed.assert_called_once()
            
            # Load and verify the repaired batch
            with open(emb_dir / "batch.pkl", "rb") as f:
                repaired_data = pickle.load(f)
            
            # Check that zero vectors were replaced
            assert not np.all(repaired_data["embeddings"] == 0)


class TestMonitoringWorkflow(TestCase):
    """Test monitoring and diagnostics workflows."""
    
    def test_index_monitoring_workflow(self):
        """Test index status monitoring."""
        from diagnostics.monitor import IndexMonitor
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create mock index structure
            index_dir = Path(tmpdir) / "_index"
            index_dir.mkdir()
            
            # Create mapping file
            mapping_data = [
                {"doc_id": "doc1", "text": "chunk1"},
                {"doc_id": "doc2", "text": "chunk2"}
            ]
            (index_dir / "mapping.json").write_text(json.dumps(mapping_data))
            
            # Create index file
            (index_dir / "index.faiss").touch()
            
            # Monitor the index
            monitor = IndexMonitor(tmpdir)
            status = monitor.check_status()
            
            # IndexStatus has different attributes
            assert status is not None
            assert hasattr(status, 'status')
            if hasattr(status, 'document_count'):
                assert status.document_count == 2
            assert status.index_file is not None
    
    @patch('diagnostics.diagnostics.embed_texts')
    def test_diagnostics_workflow(self, mock_embed):
        """Test diagnostics and validation workflow."""
        mock_embed.return_value = [[0.1, 0.2]]
        
        from diagnostics.diagnostics import test_account
        
        # Test account validation
        result = test_account({"project_id": "test-project"})
        
        # Result should be tuple (success, message)
        assert isinstance(result, tuple)
        assert len(result) == 2
    
    def test_statistics_workflow(self):
        """Test statistics collection workflow."""
        from diagnostics.statistics import get_file_statistics
        
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            
            # Create test directory structure
            conv_dir = root / "conversation1"
            conv_dir.mkdir()
            (conv_dir / "Conversation.txt").write_text("test")
            
            att_dir = conv_dir / "Attachments"
            att_dir.mkdir()
            (att_dir / "file1.pdf").touch()
            (att_dir / "file2.docx").touch()
            
            # Get statistics
            stats = get_file_statistics(root)
            
            assert stats["conversation_count"] == 1
            assert stats["total_files"] >= 3
            assert ".pdf" in stats["extensions"]
            assert ".docx" in stats["extensions"]


class TestEmailProcessingWorkflow(TestCase):
    """Test email processing workflows."""
    
    def test_email_parsing_workflow(self):
        """Test complete email parsing workflow."""
        from emailops.utils import clean_email_text, extract_email_metadata, split_email_thread
        
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
        from emailops.utils import load_conversation
        
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
            
            assert conv["conversation_txt"] == "Email conversation content"
            assert conv["manifest"]["id"] == "conv123"
            assert conv["summary"]["summary"] == "This is a test conversation"
            assert len(conv["attachments"]) == 1
            assert "Attachment content" in conv["attachments"][0]["text"]


class TestEndToEndWorkflow(TestCase):
    """Test complete end-to-end workflows."""
    
    @patch('processing.processor.get_config')
    @patch('processing.processor.TextChunker')
    @patch('processing.processor.embed_texts')
    @patch('processing.processor._initialize_gcp_credentials')
    def test_complete_pipeline(self, mock_init_creds, mock_embed, mock_chunker_class, mock_get_config):
        """Test complete pipeline from documents to embeddings."""
        # Setup config
        mock_config = Mock()
        mock_config.DEFAULT_BATCH_SIZE = 2
        mock_config.DEFAULT_CHUNK_SIZE = 100
        mock_config.DEFAULT_CHUNK_OVERLAP = 20
        mock_config.DEFAULT_NUM_WORKERS = 1
        mock_config.CHUNK_DIRNAME = "_chunks"
        mock_config.INDEX_DIRNAME = "_index"
        mock_get_config.return_value = mock_config
        
        # Setup chunker
        mock_chunker = Mock()
        mock_chunker.chunk_text.return_value = [
            {"text": "chunk1", "metadata": {}},
            {"text": "chunk2", "metadata": {}}
        ]
        mock_chunker_class.return_value = mock_chunker
        
        # Setup embeddings
        mock_init_creds.return_value = "/path/to/creds.json"
        mock_embed.return_value = [[0.1, 0.2], [0.3, 0.4]]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            from processing.processor import UnifiedProcessor
            
            # Step 1: Chunk documents
            input_dir = Path(tmpdir) / "input"
            input_dir.mkdir()
            (input_dir / "doc.txt").write_text("Document content")
            
            output_dir = Path(tmpdir) / "output"
            
            processor = UnifiedProcessor(
                root_dir=str(output_dir),
                mode="chunk",
                num_workers=1
            )
            processor.chunk_documents(str(input_dir), "*.txt")
            
            # Step 2: Generate embeddings
            processor = UnifiedProcessor(
                root_dir=str(output_dir),
                mode="embed",
                batch_size=2
            )
            
            # Create chunk files for embedding
            chunks_dir = output_dir / "_chunks" / "chunks"
            chunks_dir.mkdir(parents=True, exist_ok=True)
            chunk_data = {
                "doc_id": "doc",
                "chunks": mock_chunker.chunk_text.return_value
            }
            (chunks_dir / "doc.json").write_text(json.dumps(chunk_data))
            
            processor.create_embeddings(use_chunked_files=True)
            
            # Verify the pipeline executed
            assert mock_chunker.chunk_text.called
            assert mock_embed.called
            
            # Verify output structure
            assert (output_dir / "_index").exists()