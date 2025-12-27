import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Adjust path to import local module if needed
sys.path.append(str(Path(__file__).resolve().parents[3] / "cli/src"))

from cortex_cli.cmd_doctor import (
    _c,
    _install_packages,
    _normalize_provider,
    _probe_embeddings,
    _try_import,
    check_and_install_dependencies,
    check_exports,
    check_index_health,
    check_ingest,
    check_postgres,
    check_redis,
    check_reranker,
    main,
)


class TestCmdDoctor(unittest.TestCase):
    def setUp(self):
        # Patch sys.modules locally if needed, but do it safely
        self.modules_patcher = patch.dict(
            sys.modules,
            {
                "cortex.ingestion.parser_email": MagicMock(),
                "cortex.ingestion.text_preprocessor": MagicMock(),
            },
        )
        self.modules_patcher.start()

    def tearDown(self):
        self.modules_patcher.stop()

    @patch("sys.stdout")
    def test_colors_basic(self, mock_stdout):
        mock_stdout.isatty.return_value = True
        out = _c("test", "red")
        self.assertIn("test", out)
        self.assertIn("\033[31m", out)

        mock_stdout.isatty.return_value = False
        out2 = _c("test", "red")
        self.assertEqual(out2, "test")

    def test_normalize_provider(self):
        self.assertEqual(_normalize_provider("gcp"), "vertex")
        self.assertEqual(_normalize_provider("vertexai"), "vertex")
        self.assertEqual(_normalize_provider("openai"), "openai")

    def test_try_import(self):
        success, err = _try_import("os")
        self.assertTrue(success)
        self.assertEqual(err, "ok")

        success, err = _try_import("non_existent_module_xyz")
        self.assertFalse(success)
        self.assertEqual(err, "not_installed")

    @patch("subprocess.run")
    def test_install_packages(self, mock_run):
        mock_run.return_value.returncode = 0
        res = _install_packages(["numpy"], timeout=10)
        self.assertTrue(res)

    @patch("subprocess.run")
    def test_install_packages_fail(self, mock_run):
        mock_run.return_value.returncode = 1
        res = _install_packages(["numpy"], timeout=10)
        self.assertFalse(res)

    @patch("cortex_cli.cmd_doctor._install_packages")
    def test_check_and_install_dependencies_auto(self, mock_install):
        mock_install.return_value = True
        with patch("cortex_cli.cmd_doctor._try_import") as mock_try:
            # We need conditional return values for _try_import
            def side_effect(name):
                # We are testing check_and_install_dependencies("openai"...)
                # It calls _try_import for each package in critical/optional list.
                # "openai" is the critical one.
                # If mock_install hasn't been called yet, simulate it missing.
                if name == "openai" and not mock_install.called:
                    return False, "not_installed"
                return True, "ok"

            mock_try.side_effect = side_effect
            check_and_install_dependencies("openai", auto_install=True)
            self.assertTrue(mock_install.called)

    def test_check_postgres_success(self):
        config = MagicMock()
        config.database.url = "postgres://user:pass@host:5432/db"

        with patch("sqlalchemy.create_engine"):
            # check_postgres returns (bool, str|None)
            success, err = check_postgres(config)
            self.assertTrue(success)
            self.assertIsNone(err)

    def test_check_postgres_fail(self):
        config = MagicMock()
        config.database.url = "postgres://bad"
        with patch("sqlalchemy.create_engine") as mock_engine:
            mock_engine.return_value.connect.side_effect = Exception(
                "Connection refused"
            )
            success, err = check_postgres(config)
            self.assertFalse(success)
            self.assertIn("Connection refused", err)

    def test_check_redis_success(self):
        config = MagicMock()
        with patch("redis.from_url") as mock_redis_from_url:
            mock_redis = mock_redis_from_url.return_value
            mock_redis.ping.return_value = True
            success, _err = check_redis(config)
            self.assertTrue(success)

    def test_check_redis_fail(self):
        config = MagicMock()
        with patch("redis.from_url") as mock_redis_from_url:
            mock_redis_from_url.side_effect = Exception("Redis down")
            success, _err = check_redis(config)
            self.assertFalse(success)

    def test_check_exports_success(self):
        config = MagicMock()
        config.directories.export_root = "exports"
        mock_root = MagicMock()
        mock_export = MagicMock()
        mock_root.__truediv__.return_value = mock_export
        mock_export.exists.return_value = True

        folder = MagicMock()
        folder.is_dir.return_value = True
        folder.name = "B1"
        (folder / "manifest.json").exists.return_value = True

        mock_export.iterdir.return_value = [folder]

        success, folders, _msg = check_exports(config, mock_root)
        self.assertTrue(success)
        self.assertEqual(folders, ["B1"])

    def test_check_index_health_success(self):
        config = MagicMock()
        config.directories.index_dirname = "_index"
        config.database.url = "postgres://..."
        config.embedding.output_dimensionality = 768

        mock_root = MagicMock()
        mock_index = MagicMock()
        # Mock __truediv__ to return mock_index when called with index_dirname
        mock_root.__truediv__.side_effect = lambda x: (
            mock_index if x == "_index" else MagicMock()
        )
        mock_index.exists.return_value = True
        mock_index.rglob.return_value = ["file1", "file2"]

        # Patch cortex_cli.cmd_doctor.create_engine because it's imported at top level
        with patch("cortex_cli.cmd_doctor.create_engine") as mock_create_engine:
            # Setup fetchone return for dim check
            mock_conn = (
                mock_create_engine.return_value.connect.return_value.__enter__.return_value
            )
            # The row object returned by fetchone
            row_mock = MagicMock()
            row_mock.dim = 768
            # Make sure int(row_mock.dim) works if logic calls it, though row_mock.dim IS int
            # logic: if result and result.dim: ... int(result.dim)

            mock_conn.execute.return_value.fetchone.return_value = row_mock

            success, status, err = check_index_health(config, mock_root)
            self.assertTrue(success, f"Error: {err}")
            self.assertEqual(status["file_count"], 2)

    def test_check_ingest_parser_success(self):
        config = MagicMock()
        mock_root = MagicMock()
        mock_export = MagicMock()
        # Ensure division returns export
        mock_root.__truediv__.return_value = mock_export
        mock_export.exists.return_value = True

        folder = MagicMock()
        folder.is_dir.return_value = True
        (folder / "messages").exists.return_value = True
        eml_file = MagicMock()
        eml_file.suffix = ".eml"
        # glob needs to return iterable
        (folder / "messages").glob.side_effect = lambda pat: (
            [eml_file] if pat == "*.eml" else []
        )
        mock_export.iterdir.return_value = [folder]

        # Patch the imported module locally if needed or rely on setUp
        # We need to set the specific return value for parse_eml_file
        sys.modules[
            "cortex.ingestion.parser_email"
        ].parse_eml_file.return_value.message_id = "123"

        success, details, msg = check_ingest(config, mock_root)
        self.assertTrue(success, f"Ingest check failed: {msg}")
        self.assertTrue(details["parser_ok"])

    def test_check_reranker(self):
        config = MagicMock()
        config.search.reranker_endpoint = "http://reranker"
        with patch("httpx.get") as mock_get:
            mock_get.return_value.status_code = 200
            success, _err = check_reranker(config)
            self.assertTrue(success)

    def test_probe_embeddings(self):
        # _probe_embeddings uses delayed import
        with patch.dict(sys.modules, {"cortex.llm.client": MagicMock()}):
            sys.modules["cortex.llm.client"].embed_texts.return_value = [[0.1, 0.2]]
            success, dim = _probe_embeddings("vertex")
            self.assertTrue(success)
            self.assertEqual(dim, 2)

    @patch("cortex_cli.cmd_doctor.get_config")
    def test_main_all_pass(self, mock_get_config):
        config = MagicMock()
        mock_get_config.return_value = config

        # Setup config for various checks
        config.database.url = "postgres://user:pass@host:5432/db"
        config.embedding.output_dimensionality = (
            768  # Correct attribute for check_index_health
        )
        config.search.reranker_endpoint = "http://reranker"

        # Patch the dependencies used INSIDE the check functions, not the functions themselves
        with (
            patch(
                "cortex_cli.cmd_doctor.check_and_install_dependencies"
            ) as mock_dep,  # this one is fine to mock as we tested it separately
            patch("cortex_cli.cmd_doctor.Path") as mock_path_cls,
            patch("sqlalchemy.create_engine") as mock_engine,
            patch("redis.Redis") as mock_redis_cls,
            patch("httpx.get") as mock_http_get,
            patch(
                "sys.argv",
                [
                    "doctor",
                    "--check-index",
                    "--check-db",
                    "--check-redis",
                    "--check-exports",
                    "--check-ingest",
                    "--check-embeddings",
                    "--check-reranker",
                ],
            ),
            # Patch module-level imports used by check_index_health
            patch("cortex_cli.cmd_doctor.create_engine") as mock_create_engine_mod,
            patch("cortex_cli.cmd_doctor.text"),
            # We still need to mock some internal helpers if they satisfy dependencies
            patch("cortex_cli.cmd_doctor._probe_embeddings", return_value=(True, 768)),
            patch(
                "cortex_cli.cmd_doctor._find_sample_file",
                return_value=Path("sample.eml"),
            ),
            patch(
                "cortex_cli.cmd_doctor._test_parser_on_file",
                return_value=(True, "Subject", None),
            ),
        ):
            # Setup DB mock
            # Link module-level mock to the same engine mock configuration
            mock_create_engine_mod.return_value = mock_engine.return_value

            mock_conn = MagicMock()
            mock_engine.return_value.connect.return_value.__enter__.return_value = (
                mock_conn
            )
            mock_result = MagicMock()
            mock_conn.execute.return_value = mock_result

            # Helper for sequential calls to execute/fetchone
            mock_result.scalar.return_value = 1
            # check_index_health calls fetchone() expecting an object with .dim
            # check_db is not called by main() in this flow, so we don't need migration response
            mock_result.fetchone.return_value = MagicMock(dim=768)

            # Setup file system mocks
            mock_path_instance = mock_path_cls.return_value
            mock_root = mock_path_instance.expanduser.return_value.resolve.return_value
            mock_root.exists.return_value = True

            # Common mock for directories (index, export)
            mock_dir = mock_root.__truediv__.return_value
            mock_dir.exists.return_value = True
            mock_dir.rglob.return_value = [MagicMock()] * 5

            # Setup export/ingest structure
            mock_child = MagicMock()
            mock_child.is_dir.return_value = True
            mock_child.name = "export_1"
            (mock_child / "manifest.json").exists.return_value = True
            mock_messages = mock_child / "messages"
            mock_messages.exists.return_value = True
            mock_messages.glob.return_value = [MagicMock(suffix=".eml")]

            mock_dir.iterdir.return_value = [mock_child]

            # Setup Redis mock
            mock_redis = MagicMock()
            mock_redis_cls.from_url.return_value = mock_redis
            mock_redis.ping.return_value = True

            # Setup HTTP mock
            mock_http_get.return_value.status_code = 200

            # Setup dependency check return
            mock_dep.return_value = MagicMock(
                missing_critical=[], missing_optional=[], installed=[]
            )

            # Run main
            try:
                main()
            except SystemExit as e:
                if e.code != 0:
                    # Re-read stdout/stderr to debug
                    print(f"\nMain failed with code {e.code}. Capturing stdout...")
                    # Since we can't easily capture output already printed to sys.stdout without capsys in scope (which it is not in this method signature? wait, I can add it)
                    # Use a trick to get the reason
                    pass
                self.assertEqual(
                    e.code,
                    0,
                    "Doctor check failed with specific errors (see captured stdout)",
                )


class TestCmdDoctorExtended(unittest.TestCase):
    """Extended tests for cmd_doctor functions."""

    def test_packages_for_provider_vertex(self):
        from cortex_cli.cmd_doctor import _packages_for_provider

        critical, _optional = _packages_for_provider("vertex")
        self.assertIn("google-genai", critical)

    def test_packages_for_provider_openai(self):
        from cortex_cli.cmd_doctor import _packages_for_provider

        critical, _optional = _packages_for_provider("openai")
        self.assertIn("openai", critical)

    def test_packages_for_provider_unknown(self):
        from cortex_cli.cmd_doctor import _packages_for_provider

        critical, optional = _packages_for_provider("unknown_provider")
        # Should still return lists (possibly empty or defaults)
        self.assertIsInstance(critical, list)
        self.assertIsInstance(optional, list)

    def test_check_db_success(self):
        from cortex_cli.cmd_doctor import check_db

        config = MagicMock()
        config.database.url = "postgres://user:pass@host:5432/db"

        with patch("sqlalchemy.create_engine") as mock_engine:
            mock_conn = MagicMock()
            mock_engine.return_value.connect.return_value.__enter__ = MagicMock(
                return_value=mock_conn
            )
            mock_engine.return_value.connect.return_value.__exit__ = MagicMock()
            mock_conn.execute.return_value.scalar.return_value = 1

            success, _status, _err = check_db(config)
            self.assertTrue(success)

    def test_find_requirements_file(self):
        from cortex_cli.cmd_doctor import _find_requirements_file

        result = _find_requirements_file()
        # May return None or a Path depending on repo state
        self.assertTrue(result is None or isinstance(result, Path))

    def test_requirements_file_candidates(self):
        from cortex_cli.cmd_doctor import _requirements_file_candidates

        candidates = _requirements_file_candidates()
        self.assertIsInstance(candidates, list)
        self.assertTrue(len(candidates) > 0)

    def test_dep_report_dataclass(self):
        from cortex_cli.cmd_doctor import DepReport

        report = DepReport(
            provider="vertex",
            missing_critical=[],
            missing_optional=["faiss-cpu"],
            installed=["numpy"],
        )
        self.assertEqual(report.provider, "vertex")
        self.assertEqual(report.installed, ["numpy"])
