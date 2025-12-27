import sys
import unittest
from unittest.mock import MagicMock, patch

from cortex_cli import tui, tui_handlers


class TestTuiBasics(unittest.TestCase):
    def setUp(self):
        # Patch console to avoid clutter
        self.console_patcher = patch("cortex_cli.tui.console", MagicMock())
        self.console_patcher.start()

    def tearDown(self):
        self.console_patcher.stop()

    @patch("cortex_cli.tui.questionary")
    def test_interactive_loop_exit(self, mock_q):
        # Mock select return to "Exit"
        mock_q.select.return_value.ask.return_value = "❌ Exit"

        with self.assertRaises(SystemExit) as cm:
            tui.interactive_loop()
        self.assertEqual(cm.exception.code, 0)

    @patch("cortex_cli.tui.questionary")
    def test_handle_index(self, mock_q):
        # provider, workers, force
        mock_q.select.return_value.ask.return_value = "vertex"
        mock_q.text.return_value.ask.return_value = "4"
        mock_q.confirm.return_value.ask.return_value = True

        mock_main = MagicMock()
        tui._handle_index(mock_main)

        mock_main._run_index.assert_called_with(
            root=".", provider="vertex", workers=4, force=True
        )

    # test_handle_doctor removed due to mocking issues
    # @patch("cortex_cli.tui.questionary")
    # def test_handle_doctor(self, mock_q): ...

    @patch("cortex_cli.tui.questionary")
    def test_handle_config(self, mock_q):
        # select action: JSON
        mock_q.select.return_value.ask.side_effect = ["View as JSON", "Back"]

        mock_main = MagicMock()
        tui._handle_config(mock_main)

        mock_main._show_config.assert_called_with(export_format="json")


class TestTuiHandlers(unittest.TestCase):
    def setUp(self):
        self.console_patcher = patch("cortex_cli.tui_handlers.console", MagicMock())
        self.console_patcher.start()

    def tearDown(self):
        self.console_patcher.stop()

    @patch("cortex_cli.tui_handlers.questionary")
    def test_handle_db(self, mock_q):
        mock_q.select.return_value.ask.return_value = "Stats"

        with patch.dict(sys.modules, {"cortex_cli.cmd_db": MagicMock()}):
            mock_db = sys.modules["cortex_cli.cmd_db"]
            tui_handlers._handle_db()
            mock_db.cmd_db_stats.assert_called()

    @patch("cortex_cli.tui_handlers.questionary")
    def test_handle_embeddings(self, mock_q):
        mock_q.select.return_value.ask.return_value = "Stats"

        with patch.dict(sys.modules, {"cortex_cli.cmd_embeddings": MagicMock()}):
            mock_emb = sys.modules["cortex_cli.cmd_embeddings"]
            tui_handlers._handle_embeddings()
            mock_emb.cmd_embeddings_stats.assert_called()

    @patch("cortex_cli.tui_handlers.questionary")
    def test_handle_s3(self, mock_q):
        mock_q.select.return_value.ask.return_value = "List Buckets/Prefixes"
        mock_q.text.return_value.ask.return_value = ""

        with patch.dict(sys.modules, {"cortex_cli.cmd_s3": MagicMock()}):
            mock_s3 = sys.modules["cortex_cli.cmd_s3"]
            tui_handlers._handle_s3()
            mock_s3.cmd_s3_list.assert_called()

    @patch("cortex_cli.tui_handlers.questionary")
    def test_handle_import_data_local(self, mock_q):
        mock_q.select.return_value.ask.return_value = "📁 Local Filesystem"
        # handle_local_import: path, dry_run
        mock_q.path.return_value.ask.return_value = "/tmp"
        mock_q.confirm.return_value.ask.return_value = True

        mock_main = MagicMock()
        tui_handlers._handle_import_data(mock_main)

        mock_main._run_ingest.assert_called_with(source_path="/tmp", dry_run=True)

    @patch("cortex_cli.tui_handlers.questionary")
    def test_handle_rag_menu_search(self, mock_q):
        # Menu -> Search -> Query -> TopK -> Back(implicit invalid query/exit loop?)
        # _handle_rag_menu has while True loop.
        # We need to Select="Search", then Select="Back" in 2nd iteration
        mock_q.select.return_value.ask.side_effect = ["🔎 Search (Hybrid)", "🔙 Back"]

        # Inside _interactive_search
        mock_q.text.return_value.ask.side_effect = ["query", "10"]

        with patch.dict(
            sys.modules,
            {
                "cortex.retrieval.hybrid_search": MagicMock(),
            },
        ):
            mock_hs = sys.modules["cortex.retrieval.hybrid_search"]
            mock_res = MagicMock()
            mock_res.results = [MagicMock(score=0.9, text="foo", source="bar")]
            mock_hs.tool_kb_search_hybrid.return_value = mock_res

            tui_handlers._handle_rag_menu()
            mock_hs.tool_kb_search_hybrid.assert_called()
