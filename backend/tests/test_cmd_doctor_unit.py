from unittest.mock import MagicMock, patch

import pytest
from cortex_cli.cmd_doctor import (
    _find_sample_conversation_dir,
    _test_loader_on_dir,
    _test_preprocessor_import,
)


class TestCmdDoctorHelpers:
    @pytest.fixture
    def mock_export_root(self, tmp_path):
        root = tmp_path / "exports"
        root.mkdir()
        return root

    def test_find_sample_conversation_dir_no_root(self, tmp_path):
        assert _find_sample_conversation_dir(tmp_path / "nonexistent") is None

    def test_find_sample_conversation_dir_success(self, mock_export_root):
        folder = mock_export_root / "folder1"
        folder.mkdir()
        manifest = folder / "manifest.json"
        manifest.touch()

        found = _find_sample_conversation_dir(mock_export_root)
        assert found == folder

    def test_find_sample_conversation_dir_none(self, mock_export_root):
        folder = mock_export_root / "folder1"
        folder.mkdir()
        # No manifest.json
        assert _find_sample_conversation_dir(mock_export_root) is None

    @patch("cortex.ingestion.conv_loader.load_conversation")
    @patch("cortex.ingestion.core_manifest.resolve_subject")
    def test_test_loader_on_dir_success(
        self, mock_resolve_subject, mock_load_conversation, tmp_path
    ):
        sample_dir = tmp_path / "sample_convo"
        sample_dir.mkdir()
        mock_load_conversation.return_value = {
            "manifest": {"subject": "Test"},
            "summary": {},
        }
        mock_resolve_subject.return_value = ("Test Subject", "test subject")

        success, subject, error = _test_loader_on_dir(sample_dir)
        assert success is True
        assert subject == "Test Subject"
        assert error is None
        mock_load_conversation.assert_called_once_with(sample_dir)
        mock_resolve_subject.assert_called_once()

    @patch("cortex.ingestion.conv_loader.load_conversation")
    def test_test_loader_on_dir_fail_load(self, mock_load_conversation, tmp_path):
        sample_dir = tmp_path / "sample_convo"
        sample_dir.mkdir()
        mock_load_conversation.return_value = None  # Loader failed

        success, subject, error = _test_loader_on_dir(sample_dir)
        assert success is False
        assert subject is None
        assert error == "load_conversation returned no data"

    def test_test_loader_on_dir_import_error(self, tmp_path):
        sample_dir = tmp_path / "sample_convo"
        sample_dir.mkdir()
        with patch.dict("sys.modules", {"cortex.ingestion.conv_loader": None}):
            success, subject, error = _test_loader_on_dir(sample_dir)
            assert success is False
            assert error == "Failed to import conversation loader"

    def test_test_preprocessor_import_success(self):
        with patch.dict(
            "sys.modules", {"cortex.ingestion.text_preprocessor": MagicMock()}
        ):
            success, error = _test_preprocessor_import()
            assert success is True
            assert error is None

    def test_test_preprocessor_import_fail(self):
        with patch.dict("sys.modules", {"cortex.ingestion.text_preprocessor": None}):
            success, error = _test_preprocessor_import()
            assert success is False
            assert error == "Failed to import text preprocessor"
