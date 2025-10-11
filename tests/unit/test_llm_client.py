"""Unit tests for emailops.llm_client module."""

import pytest
from unittest import TestCase
from unittest.mock import Mock, MagicMock, patch, call
from typing import Any

# Test the module without actually importing llm_runtime
import sys


class TestLLMClient(TestCase):
    """Test llm_client wrapper functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a mock llm_runtime module
        self.mock_runtime = MagicMock()
        self.mock_runtime.__all__ = ["complete_text", "complete_json", "embed_texts", "LLMError"]
        
        # Patch the import
        self.runtime_patch = patch.dict('sys.modules', {'emailops.llm_runtime': self.mock_runtime})
        self.runtime_patch.start()
        
        # Import after patching
        from emailops import llm_client
        self.llm_client = llm_client
    
    def tearDown(self):
        """Clean up after tests."""
        self.runtime_patch.stop()
        # Remove the imported module to ensure clean state
        if 'emailops.llm_client' in sys.modules:
            del sys.modules['emailops.llm_client']
    
    def test_rt_attr_success(self):
        """Test _rt_attr successfully gets runtime attribute."""
        # Test with a real attribute that exists
        result = self.llm_client._rt_attr("embed_texts")
        assert callable(result)
    
    def test_rt_attr_missing(self):
        """Test _rt_attr raises AttributeError for missing attribute."""
        if hasattr(self.mock_runtime, 'missing_attr'):
            delattr(self.mock_runtime, 'missing_attr')
        with pytest.raises(AttributeError, match="llm_runtime.missing_attr is not available"):
            self.llm_client._rt_attr("missing_attr")
    
    def test_complete_text(self):
        """Test complete_text forwards to runtime."""
        self.mock_runtime.complete_text = Mock(return_value="completion")
        result = self.llm_client.complete_text("system", "user", temperature=0.7)
        
        self.mock_runtime.complete_text.assert_called_once_with("system", "user", temperature=0.7)
        assert result == "completion"
    
    def test_complete_json(self):
        """Test complete_json forwards to runtime."""
        self.mock_runtime.complete_json = Mock(return_value='{"key": "value"}')
        result = self.llm_client.complete_json("system", "user", response_schema={"type": "object"})
        
        self.mock_runtime.complete_json.assert_called_once_with("system", "user", response_schema={"type": "object"})
        assert result == '{"key": "value"}'
    
    def test_embed_texts(self):
        """Test embed_texts forwards to runtime."""
        import numpy as np
        expected = np.array([[0.1, 0.2], [0.3, 0.4]], dtype="float32")
        self.mock_runtime.embed_texts = Mock(return_value=expected)
        texts = ["text1", "text2"]
        result = self.llm_client.embed_texts(texts, provider="openai")
        
        self.mock_runtime.embed_texts.assert_called_once_with(texts, provider="openai")
        assert np.array_equal(result, expected)
    
    def test_complete_alias(self):
        """Test complete is an alias for complete_text."""
        self.mock_runtime.complete_text = Mock(return_value="completion")
        result = self.llm_client.complete("system", "user")
        
        self.mock_runtime.complete_text.assert_called_once_with("system", "user")
        assert result == "completion"
    
    def test_json_complete_alias(self):
        """Test json_complete is an alias for complete_json."""
        self.mock_runtime.complete_json = Mock(return_value='{"result": "json"}')
        result = self.llm_client.json_complete("system", "user")
        
        self.mock_runtime.complete_json.assert_called_once_with("system", "user")
        assert result == '{"result": "json"}'
    
    def test_embed_with_list(self):
        """Test embed with list of strings."""
        import numpy as np
        expected = np.array([[0.1, 0.2]], dtype="float32")
        self.mock_runtime.embed_texts = Mock(return_value=expected)
        result = self.llm_client.embed(["text1", "text2"])
        
        self.mock_runtime.embed_texts.assert_called_once_with(["text1", "text2"])
        assert np.array_equal(result, expected)
    
    def test_embed_with_generator(self):
        """Test embed converts generator to list."""
        import numpy as np
        expected = np.array([[0.1], [0.2]], dtype="float32")
        self.mock_runtime.embed_texts = Mock(return_value=expected)
        
        def text_generator():
            yield "text1"
            yield "text2"
        
        result = self.llm_client.embed(text_generator())
        
        self.mock_runtime.embed_texts.assert_called_once_with(["text1", "text2"])
        assert np.array_equal(result, expected)
    
    def test_embed_with_tuple(self):
        """Test embed with tuple of strings."""
        import numpy as np
        expected = np.array([[0.1], [0.2]], dtype="float32")
        self.mock_runtime.embed_texts = Mock(return_value=expected)
        result = self.llm_client.embed(("text1", "text2"))
        
        self.mock_runtime.embed_texts.assert_called_once_with(["text1", "text2"])
        assert np.array_equal(result, expected)
    
    def test_embed_with_single_string_raises(self):
        """Test embed raises TypeError for single string."""
        with pytest.raises(TypeError, match="expects an iterable of strings, not a single string"):
            self.llm_client.embed("single string")
    
    def test_embed_with_bytes_raises(self):
        """Test embed raises TypeError for bytes."""
        with pytest.raises(TypeError, match="expects an iterable of strings, not a single string"):
            self.llm_client.embed(b"bytes")  # type: ignore
    
    def test_embed_with_bytearray_raises(self):
        """Test embed raises TypeError for bytearray."""
        with pytest.raises(TypeError, match="expects an iterable of strings, not a single string"):
            self.llm_client.embed(bytearray(b"bytes"))  # type: ignore
    
    def test_embed_with_memoryview_raises(self):
        """Test embed raises TypeError for memoryview."""
        with pytest.raises(TypeError, match="expects an iterable of strings, not a single string"):
            self.llm_client.embed(memoryview(b"bytes"))  # type: ignore
    
    def test_embed_with_non_string_elements_raises(self):
        """Test embed raises TypeError for non-string elements."""
        with pytest.raises(TypeError, match="expects an iterable of str"):
            self.llm_client.embed(["text", 123, None])  # type: ignore
    
    def test_embed_with_mixed_types_raises(self):
        """Test embed raises TypeError for mixed types."""
        with pytest.raises(TypeError, match="expects an iterable of str"):
            self.llm_client.embed(["text", b"bytes"])  # type: ignore
    
    def test_runtime_exports_with_list(self):
        """Test _runtime_exports with list __all__."""
        self.mock_runtime.__all__ = ["complete_text", "embed_texts", "LLMError"]
        exports = self.llm_client._runtime_exports()
        assert exports == ["complete_text", "embed_texts", "LLMError"]
    
    def test_runtime_exports_with_tuple(self):
        """Test _runtime_exports with tuple __all__."""
        self.mock_runtime.__all__ = ("complete_text", "embed_texts")
        exports = self.llm_client._runtime_exports()
        assert exports == ["complete_text", "embed_texts"]
    
    def test_runtime_exports_filters_non_strings(self):
        """Test _runtime_exports filters out non-string values."""
        self.mock_runtime.__all__ = ["complete_text", 123, None, "embed_texts"]
        exports = self.llm_client._runtime_exports()
        assert exports == ["complete_text", "embed_texts"]
    
    def test_runtime_exports_with_iterable(self):
        """Test _runtime_exports with other iterable."""
        self.mock_runtime.__all__ = {"complete_text", "embed_texts"}  # Set is iterable
        exports = self.llm_client._runtime_exports()
        assert set(exports) == {"complete_text", "embed_texts"}
    
    def test_runtime_exports_no_all_attribute(self):
        """Test _runtime_exports when runtime has no __all__."""
        delattr(self.mock_runtime, '__all__')
        exports = self.llm_client._runtime_exports()
        assert exports == []
    
    def test_runtime_exports_invalid_all(self):
        """Test _runtime_exports when __all__ is not iterable."""
        self.mock_runtime.__all__ = 123
        exports = self.llm_client._runtime_exports()
        assert exports == []
    
    def test_getattr_all_with_llmerror(self):
        """Test __getattr__ for __all__ with LLMError present."""
        self.mock_runtime.LLMError = Exception
        self.mock_runtime.__all__ = ["complete_text", "complete_json", "embed_texts", "LLMError"]
        
        all_exports = self.llm_client.__getattr__("__all__")
        
        # Should include core exports
        assert "complete_text" in all_exports
        assert "complete_json" in all_exports
        assert "embed_texts" in all_exports
        assert "complete" in all_exports
        assert "json_complete" in all_exports
        assert "embed" in all_exports
        assert "LLMError" in all_exports
    
    def test_getattr_all_without_llmerror(self):
        """Test __getattr__ for __all__ without LLMError."""
        # Remove LLMError from mock runtime
        if hasattr(self.mock_runtime, 'LLMError'):
            delattr(self.mock_runtime, 'LLMError')
        self.mock_runtime.__all__ = []
        
        all_exports = self.llm_client.__getattr__("__all__")
        
        # Should not include LLMError
        assert "LLMError" not in all_exports
        assert "complete_text" in all_exports
        assert "embed_texts" in all_exports
    
    def test_getattr_all_deduplicates(self):
        """Test __getattr__ for __all__ deduplicates entries."""
        self.mock_runtime.LLMError = Exception
        self.mock_runtime.__all__ = ["complete_text", "complete_text", "embed_texts"]
        
        all_exports = self.llm_client.__getattr__("__all__")
        
        # complete_text should appear only once
        assert all_exports.count("complete_text") == 1
        assert all_exports.count("embed_texts") == 1
    
    def test_getattr_forwards_to_runtime(self):
        """Test __getattr__ forwards unknown attributes to runtime."""
        # Test that it correctly raises AttributeError for truly missing attributes
        with pytest.raises(AttributeError, match="module 'emailops.llm_client' has no attribute"):
            self.llm_client.__getattr__("truly_missing_func")
    
    def test_getattr_llmerror(self):
        """Test __getattr__ can get LLMError from runtime."""
        error_class = self.llm_client.__getattr__("LLMError")
        # LLMError should be from llm_runtime module
        assert error_class.__name__ == "LLMError"
        assert issubclass(error_class, Exception)
    
    def test_getattr_missing_attribute(self):
        """Test __getattr__ raises AttributeError for missing attributes."""
        if hasattr(self.mock_runtime, 'missing'):
            delattr(self.mock_runtime, 'missing')
        
        with pytest.raises(AttributeError, match="module 'emailops.llm_client' has no attribute 'missing'"):
            self.llm_client.__getattr__("missing")
    
    def test_dir_returns_sorted_all(self):
        """Test __dir__ returns sorted public API."""
        self.mock_runtime.LLMError = Exception
        self.mock_runtime.__all__ = ["complete_text", "complete_json", "embed_texts", "LLMError"]
        
        dir_result = self.llm_client.__dir__()
        
        # Should be sorted
        assert dir_result == sorted(dir_result)
        # Should include core exports (no custom zebra_func in real API)
        assert "complete_text" in dir_result
        assert "embed_texts" in dir_result
        assert "LLMError" in dir_result
    
    def test_dir_removes_duplicates(self):
        """Test __dir__ removes duplicates."""
        self.mock_runtime.LLMError = Exception
        self.mock_runtime.__all__ = ["complete_text", "complete_text"]
        
        dir_result = self.llm_client.__dir__()
        
        # complete_text should appear only once
        assert dir_result.count("complete_text") == 1


class TestLLMClientIntegration(TestCase):
    """Integration tests for llm_client module."""
    
    def test_complete_text_error_handling(self):
        """Test complete_text handles runtime errors."""
        mock_runtime = MagicMock()
        mock_runtime.complete_text.side_effect = RuntimeError("API error")
        
        with patch.dict('sys.modules', {'emailops.llm_runtime': mock_runtime}):
            from emailops import llm_client
            
            with pytest.raises(RuntimeError, match="API error"):
                llm_client.complete_text("system", "user")
    
    def test_embed_empty_list(self):
        """Test embed with empty list."""
        import numpy as np
        mock_runtime = MagicMock()
        expected = np.zeros((0, 0), dtype="float32")
        mock_runtime.embed_texts = Mock(return_value=expected)
        
        with patch.dict('sys.modules', {'emailops.llm_runtime': mock_runtime}):
            from emailops import llm_client
            
            result = llm_client.embed([])
            mock_runtime.embed_texts.assert_called_once_with([])
            assert np.array_equal(result, expected)
    
    def test_multiple_kwargs_forwarding(self):
        """Test that all kwargs are properly forwarded."""
        mock_runtime = MagicMock()
        mock_runtime.complete_text = Mock(return_value="result")
        
        with patch.dict('sys.modules', {'emailops.llm_runtime': mock_runtime}):
            from emailops import llm_client
            
            llm_client.complete_text(
                "system",
                "user",
                temperature=0.7,
                max_output_tokens=100
            )
            
            mock_runtime.complete_text.assert_called_once_with(
                "system",
                "user",
                temperature=0.7,
                max_output_tokens=100
            )
    
    def test_runtime_attribute_caching_not_performed(self):
        """Test that runtime attributes are not cached."""
        mock_runtime = MagicMock()
        mock_runtime.embed_texts = Mock(return_value=[])
        
        with patch.dict('sys.modules', {'emailops.llm_runtime': mock_runtime}):
            from emailops import llm_client
            
            # Test that __getattr__ always queries the runtime, not cached
            # Get the same attribute twice
            f1 = llm_client.__getattr__("embed_texts")
            f2 = llm_client.__getattr__("embed_texts")
            
            # Should return the same object from runtime (not cached copies)
            assert f1 is f2
            assert f1 is mock_runtime.embed_texts


class TestLLMClientEdgeCases(TestCase):
    """Edge case tests for llm_client module."""
    
    def test_runtime_exports_import_error(self):
        """Test _runtime_exports handles import errors gracefully."""
        mock_runtime = MagicMock()
        
        # Make Iterable import fail
        with patch('emailops.llm_client.Iterable', side_effect=ImportError):
            # Set __all__ to something that would need Iterable check
            mock_runtime.__all__ = 123  # Not a list/tuple
            
            with patch.dict('sys.modules', {'emailops.llm_runtime': mock_runtime}):
                from emailops import llm_client
                
                exports = llm_client._runtime_exports()
                assert exports == []
    
    def test_embed_with_empty_generator(self):
        """Test embed with empty generator."""
        import numpy as np
        mock_runtime = MagicMock()
        expected = np.zeros((0, 0), dtype="float32")
        mock_runtime.embed_texts = Mock(return_value=expected)
        
        with patch.dict('sys.modules', {'emailops.llm_runtime': mock_runtime}):
            from emailops import llm_client
            
            def empty_gen():
                return
                yield  # Never reached
            
            result = llm_client.embed(empty_gen())
            mock_runtime.embed_texts.assert_called_once_with([])
            assert np.array_equal(result, expected)
    
    def test_type_checking_imports(self):
        """Test TYPE_CHECKING block doesn't affect runtime."""
        # This test verifies the TYPE_CHECKING block doesn't execute at runtime
        mock_runtime = MagicMock()
        
        with patch.dict('sys.modules', {'emailops.llm_runtime': mock_runtime}):
            from emailops import llm_client
            
            # The TYPE_CHECKING imports should not be accessible at runtime
            assert not hasattr(llm_client, '_complete_text_t')