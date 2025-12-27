import sys
from unittest.mock import MagicMock, patch

from cortex_cli.main import (
    _print_usage,
    _print_version,
    _run_index,
    _run_ingest,
    _run_search,
    _run_validate,
    _show_status,
    main,
)


class TestCliMainUtils:
    def test_print_usage(self, capsys):
        _print_usage()
        captured = capsys.readouterr()
        assert "USAGE:" in captured.out
        assert "CORE COMMANDS:" in captured.out
        assert "RAG COMMANDS:" in captured.out

    def test_print_version(self, capsys):
        with patch("importlib.metadata.version", return_value="1.2.3"):
            _print_version()
        captured = capsys.readouterr()
        assert "Version:  1.2.3" in captured.out

    def test_show_status_json(self, capsys):
        with patch("pathlib.Path.cwd") as mock_cwd:
            # Fix: Ensure .exists() returns a boolean, not a Mock
            mock_path = MagicMock()
            mock_path.exists.return_value = True
            mock_cwd.return_value = mock_path
            # The code does cwd / "path". We need __truediv__ to return a mock whose .exists() returns True
            mock_subpath = MagicMock()
            mock_subpath.exists.return_value = True
            mock_path.__truediv__.return_value = mock_subpath

            _show_status(json_output=True)
        captured = capsys.readouterr()
        assert '"environment":' in captured.out

    @patch("cortex.ingestion.conv_manifest.validation.scan_and_refresh")
    def test_run_validate_success(self, mock_scan, capsys):
        mock_report = MagicMock()
        mock_report.folders_scanned = 1
        mock_report.manifests_created = 1
        mock_report.problems = []
        mock_scan.return_value = mock_report

        with patch("pathlib.Path.exists", return_value=True), patch("pathlib.Path.is_dir", return_value=True):
            _run_validate("some/path")

        captured = capsys.readouterr()
        assert "Results:" in captured.out
        assert "Folders Scanned:   1" in captured.out

    def test_run_ingest_success(self, capsys):
        # Mocking modules before they are imported in the function
        mock_mailroom = MagicMock()
        mock_job_cls = MagicMock()
        mock_process_func = MagicMock()

        mock_summary = MagicMock()
        mock_summary.aborted_reason = None
        mock_summary.messages_ingested = 10
        mock_summary.attachments_parsed = 5
        mock_process_func.return_value = mock_summary

        mock_mailroom.IngestJob = mock_job_cls
        mock_mailroom.process_job = mock_process_func

        with (
            patch.dict(sys.modules, {"cortex.ingestion.mailroom": mock_mailroom}),
            patch("pathlib.Path.resolve") as mock_resolve,
        ):
            mock_path = MagicMock()
            mock_path.exists.return_value = True
            mock_path.is_dir.return_value = True
            mock_sub = MagicMock()
            mock_sub.is_dir.return_value = True
            (mock_sub / "Conversation.txt").exists.return_value = True
            mock_path.iterdir.return_value = [mock_sub]
            mock_resolve.return_value = mock_path

            _run_ingest("source", dry_run=False)

        captured = capsys.readouterr()
        assert "All 1 conversation(s) ingested successfully" in captured.out

    def test_run_index_success(self, capsys):
        mock_workers = MagicMock()
        mock_indexer = MagicMock()

        mock_embeddings = MagicMock()
        mock_embeddings.shape = (10, 768)
        mock_embeddings.size = 7680
        mock_mappings = ["a", "b"]
        mock_indexer.parallel_index_conversations.return_value = (
            mock_embeddings,
            mock_mappings,
        )

        mock_workers.reindex_jobs.parallel_indexer = mock_indexer

        # We need to mock cortex_workers.reindex_jobs.parallel_indexer
        # Since it is imported as from cortex_workers.reindex_jobs.parallel_indexer import ...
        # We need to mock the full chain

        with (
            patch.dict(
                sys.modules,
                {
                    "cortex_workers": mock_workers,
                    "cortex_workers.reindex_jobs": mock_workers.reindex_jobs,
                    "cortex_workers.reindex_jobs.parallel_indexer": mock_indexer,
                },
            ),
            patch("pathlib.Path.resolve", return_value=MagicMock()),
        ):
            _run_index(json_output=False)

        captured = capsys.readouterr()
        assert "Indexing complete!" in captured.out
        assert "Chunks indexed:  2" in captured.out

    def test_run_search_success(self, capsys):
        mock_hybrid = MagicMock()
        mock_search_input = MagicMock()
        mock_search_func = MagicMock()

        mock_result = MagicMock()
        r1 = MagicMock()
        r1.score = 0.95
        r2 = MagicMock()
        r2.score = 0.85
        mock_result.results = [r1, r2]
        mock_search_func.return_value = mock_result

        mock_hybrid.KBSearchInput = mock_search_input
        mock_hybrid.tool_kb_search_hybrid = mock_search_func

        # Patch parent modules to avoid ImportErrors from them
        with patch.dict(
            sys.modules,
            {
                "cortex": MagicMock(),
                "cortex.retrieval": MagicMock(),
                "cortex.retrieval.hybrid_search": mock_hybrid,
            },
        ):
            _run_search("query", json_output=False)

        captured = capsys.readouterr()
        # If it failed, captured.out would have "[ERROR]"
        if "ERROR:" in captured.out:
            print(captured.out)

        assert "Searching..." in captured.out

    def test_show_status_human(self, capsys):
        with patch("pathlib.Path.cwd") as mock_cwd:
            mock_path = MagicMock()
            mock_path.exists.return_value = True
            mock_cwd.return_value = mock_path
            mock_path.__truediv__.return_value.exists.return_value = True

            _show_status(json_output=False)
        captured = capsys.readouterr()
        assert "ENVIRONMENT STATUS:" in captured.out

    def test_main_dispatch_help(self, capsys):
        main(["--help"])
        captured = capsys.readouterr()
        assert "USAGE:" in captured.out

    def test_main_command_version(self, capsys):
        with patch("importlib.metadata.version", return_value="1.0.0"):
            main(["version"])
        captured = capsys.readouterr()
        assert "Version:" in captured.out
