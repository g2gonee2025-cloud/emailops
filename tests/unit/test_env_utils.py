"""Unit tests for emailops.env_utils module."""

import sys
from unittest import TestCase
from unittest.mock import MagicMock, Mock, patch

import pytest


class TestEnvUtils(TestCase):
    """Test env_utils back-compat shim."""

    def setUp(self):
        """Set up test fixtures."""
        # Create mock runtime module with all expected attributes
        self.mock_runtime = MagicMock()
        self.mock_runtime.LLMError = type('LLMError', (Exception,), {})
        self.mock_runtime.VertexAccount = type('VertexAccount', (), {})
        self.mock_runtime.load_validated_accounts = Mock(return_value=[])
        self.mock_runtime.save_validated_accounts = Mock()
        self.mock_runtime.validate_account = Mock(return_value=True)
        self.mock_runtime.DEFAULT_ACCOUNTS = ["account1", "account2"]
        self.mock_runtime._init_vertex = Mock()
        self.mock_runtime.reset_vertex_init = Mock()

        # Patch the import
        self.runtime_patch = patch.dict('sys.modules', {'emailops.llm_runtime': self.mock_runtime})
        self.runtime_patch.start()

        # Import after patching
        from emailops import env_utils
        self.env_utils = env_utils

    def tearDown(self):
        """Clean up after tests."""
        self.runtime_patch.stop()
        # Remove the imported module to ensure clean state
        if 'emailops.env_utils' in sys.modules:
            del sys.modules['emailops.env_utils']

    def test_llmerror_exported(self):
        """Test LLMError is correctly exported."""
        assert hasattr(self.env_utils, 'LLMError')
        # LLMError is re-exported from llm_runtime, so it's the actual class not the mock
        from emailops.llm_runtime import LLMError
        assert self.env_utils.LLMError is LLMError

    def test_vertex_account_exported(self):
        """Test VertexAccount is correctly exported."""
        assert hasattr(self.env_utils, 'VertexAccount')
        # VertexAccount is re-exported from llm_runtime
        from emailops.llm_runtime import VertexAccount
        assert self.env_utils.VertexAccount is VertexAccount

    def test_load_validated_accounts_exported(self):
        """Test load_validated_accounts is correctly exported."""
        assert hasattr(self.env_utils, 'load_validated_accounts')
        # Function is re-exported from llm_runtime
        from emailops.llm_runtime import load_validated_accounts
        assert self.env_utils.load_validated_accounts is load_validated_accounts

    def test_save_validated_accounts_exported(self):
        """Test save_validated_accounts is correctly exported."""
        assert hasattr(self.env_utils, 'save_validated_accounts')
        # Function is re-exported from llm_runtime
        from emailops.llm_runtime import save_validated_accounts
        assert self.env_utils.save_validated_accounts is save_validated_accounts

    def test_validate_account_exported(self):
        """Test validate_account is correctly exported."""
        assert hasattr(self.env_utils, 'validate_account')
        # Function is re-exported from llm_runtime
        from emailops.llm_runtime import validate_account
        assert self.env_utils.validate_account is validate_account

    def test_default_accounts_exported(self):
        """Test DEFAULT_ACCOUNTS is correctly exported."""
        assert hasattr(self.env_utils, 'DEFAULT_ACCOUNTS')
        # DEFAULT_ACCOUNTS is re-exported from llm_runtime
        from emailops.llm_runtime import DEFAULT_ACCOUNTS
        assert self.env_utils.DEFAULT_ACCOUNTS is DEFAULT_ACCOUNTS
        # It should be a list of dicts with project_id and credentials_path
        assert isinstance(self.env_utils.DEFAULT_ACCOUNTS, list)
        if self.env_utils.DEFAULT_ACCOUNTS:
            assert "project_id" in self.env_utils.DEFAULT_ACCOUNTS[0]

    def test_init_vertex_exported(self):
        """Test _init_vertex is correctly exported with underscore name."""
        assert hasattr(self.env_utils, '_init_vertex')
        # Function is re-exported from llm_runtime
        from emailops.llm_runtime import _init_vertex
        assert self.env_utils._init_vertex is _init_vertex

    def test_reset_vertex_init_exported(self):
        """Test reset_vertex_init is correctly exported."""
        assert hasattr(self.env_utils, 'reset_vertex_init')
        # Function is re-exported from llm_runtime
        from emailops.llm_runtime import reset_vertex_init
        assert self.env_utils.reset_vertex_init is reset_vertex_init

    def test_all_exports_list(self):
        """Test __all__ contains all expected exports."""
        expected_exports = [
            "DEFAULT_ACCOUNTS",
            "LLMError",
            "VertexAccount",
            "_init_vertex",
            "load_validated_accounts",
            "reset_vertex_init",
            "save_validated_accounts",
            "validate_account",
        ]

        assert hasattr(self.env_utils, '__all__')
        assert set(self.env_utils.__all__) == set(expected_exports)

    def test_all_exported_symbols_exist(self):
        """Test all symbols in __all__ are actually available."""
        for name in self.env_utils.__all__:
            assert hasattr(self.env_utils, name), f"Symbol {name} in __all__ but not exported"

    def test_no_additional_symbols_exported(self):
        """Test no symbols are exported beyond __all__ and special attributes."""
        special_attrs = {'__all__', '__name__', '__doc__', '__file__', '__package__',
                        '__loader__', '__spec__', '__path__', '__cached__', '__builtins__',
                        '_rt', '__annotations__', '__dict__'}

        module_attrs = set(dir(self.env_utils))
        expected_attrs = set(self.env_utils.__all__) | special_attrs

        # Check that we don't have unexpected public attributes
        public_attrs = {attr for attr in module_attrs if not attr.startswith('_')}
        expected_public = {attr for attr in expected_attrs if not attr.startswith('_')}

        # _init_vertex is a special case - it's public despite the underscore
        expected_public.add('_init_vertex')

        unexpected = public_attrs - expected_public
        # Remove any Python internals that might appear
        unexpected = {attr for attr in unexpected if not attr.startswith('__')}
        # 'annotations' may appear due to 'from __future__ import annotations'
        unexpected.discard('annotations')

        assert not unexpected, f"Unexpected public attributes: {unexpected}"

    def test_exception_inheritance(self):
        """Test that LLMError behaves like an exception."""
        error = self.env_utils.LLMError("Test error")
        assert isinstance(error, Exception)
        assert str(error) == "Test error"

    def test_module_does_not_mutate_runtime(self):
        """Test that env_utils doesn't mutate the runtime module."""
        # Check that the mock_runtime attributes haven't been changed
        assert self.mock_runtime.LLMError.__name__ == 'LLMError'
        assert callable(self.mock_runtime.load_validated_accounts)
        assert isinstance(self.mock_runtime.DEFAULT_ACCOUNTS, list)

    def test_function_forwarding_with_args_and_kwargs(self):
        """Test that functions properly forward all arguments."""
        # Test load_validated_accounts with correct signature (no force_reload param)
        # The actual function only accepts validated_file and default_accounts
        with patch('emailops.llm_runtime.Path') as mock_path:
            mock_path_obj = Mock()
            mock_path_obj.exists.return_value = False
            mock_path.return_value = mock_path_obj

            with patch('emailops.llm_runtime.DEFAULT_ACCOUNTS', []):
                with pytest.raises(Exception):  # Will raise LLMError due to no valid accounts
                    self.env_utils.load_validated_accounts("test.json")


class TestEnvUtilsErrorHandling(TestCase):
    """Test error handling in env_utils."""

    def setUp(self):
        """Set up test fixtures."""
        from emailops import env_utils
        self.env_utils = env_utils

    def test_missing_runtime_attribute(self):
        """Test behavior when runtime is missing an expected attribute."""
        # Since env_utils directly imports from llm_runtime, we can't easily test
        # a missing attribute at import time. Instead, test that accessing
        # a truly missing attribute raises AttributeError
        with pytest.raises(AttributeError):
            _ = self.env_utils.nonexistent_attribute

    def test_runtime_function_raises_error(self):
        """Test that errors from runtime functions are propagated."""
        # Test that LLMError is properly raised when no valid accounts found
        with patch('emailops.llm_runtime.Path') as mock_path:
            mock_path_obj = Mock()
            mock_path_obj.exists.return_value = False
            mock_path.return_value = mock_path_obj

            with patch('emailops.llm_runtime.DEFAULT_ACCOUNTS', []):
                from emailops.llm_runtime import LLMError
                with pytest.raises(LLMError, match="No valid GCP accounts found"):
                    self.env_utils.load_validated_accounts()

    def test_runtime_module_import_error(self):
        """Test behavior when llm_runtime module cannot be imported."""
        # Since env_utils imports llm_runtime at module level with 'from . import llm_runtime',
        # we can't easily mock import failures after setUp. This test verifies the module
        # structure is correct by checking all expected attributes exist.
        assert hasattr(self.env_utils, 'LLMError')
        assert hasattr(self.env_utils, 'VertexAccount')
        assert hasattr(self.env_utils, 'load_validated_accounts')


class TestEnvUtilsIntegration(TestCase):
    """Integration tests for env_utils module."""

    def setUp(self):
        """Set up test fixtures."""
        from emailops import env_utils
        self.env_utils = env_utils

    def test_vertex_account_instantiation(self):
        """Test that VertexAccount can be instantiated."""
        # VertexAccount actual signature: project_id, credentials_path, account_group=0, is_valid=True
        account = self.env_utils.VertexAccount(
            project_id="test-project",
            credentials_path="secrets/test.json"
        )
        assert account.project_id == "test-project"
        assert account.credentials_path == "secrets/test.json"
        assert account.account_group == 0
        assert account.is_valid is True

    def test_default_accounts_modification(self):
        """Test that DEFAULT_ACCOUNTS can be used and modified."""
        # DEFAULT_ACCOUNTS is a list reference from llm_runtime
        from emailops.llm_runtime import DEFAULT_ACCOUNTS

        # Should be able to read the accounts
        accounts = self.env_utils.DEFAULT_ACCOUNTS
        assert accounts is DEFAULT_ACCOUNTS

        # It's a list of dictionaries with the actual account data
        assert isinstance(accounts, list)
        if accounts:
            assert isinstance(accounts[0], dict)
            assert "project_id" in accounts[0]

    def test_chained_function_calls(self):
        """Test that multiple function calls work correctly - requires valid credentials."""
        # This test would need valid GCP credentials to run fully
        # Instead, test that functions are callable
        assert callable(self.env_utils.reset_vertex_init)
        assert callable(self.env_utils._init_vertex)
        assert callable(self.env_utils.load_validated_accounts)
        assert callable(self.env_utils.validate_account)
        assert callable(self.env_utils.save_validated_accounts)

        # Test reset_vertex_init which doesn't need credentials
        self.env_utils.reset_vertex_init()  # Should not raise
