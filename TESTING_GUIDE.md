# EmailOps Testing Guide

## Quick Start

### Install Dependencies

```bash
# Install all dependencies including testing tools
pip install -r requirements.txt
```

### Run All Tests

```bash
# Run complete test suite
pytest

# Run with coverage report
pytest --cov=. --cov-report=html

# Run specific test categories
pytest -m unit                    # Unit tests only
pytest -m integration             # Integration tests only
pytest -m "not slow"              # Skip slow tests
```

### View Coverage Report

```bash
# Generate HTML coverage report
pytest --cov=. --cov-report=html

# Open in browser (Windows)
start htmlcov/index.html

# Open in browser (macOS)
open htmlcov/index.html

# Open in browser (Linux)
xdg-open htmlcov/index.html
```

---

## Test Suite Structure

```
tests/
├── conftest.py                    # Shared fixtures and utilities
├── unit/                          # Unit tests (fast, isolated)
│   ├── test_utils.py             # diagnostics/utils.py tests
│   ├── test_diagnostics.py       # diagnostics/diagnostics.py tests
│   ├── test_monitor.py           # diagnostics/monitor.py tests
│   └── test_statistics.py        # diagnostics/statistics.py tests
├── integration/                   # Integration tests (slower)
│   ├── test_index_workflow.py    # Index creation workflow
│   └── test_embedding_pipeline.py # Embedding generation workflow
└── test_data/                     # Test fixtures and data
```

---

## Running Tests

### Basic Commands

```bash
# Run all tests with verbose output
pytest -v

# Run tests and stop at first failure
pytest -x

# Run tests matching a pattern
pytest -k "test_format_timestamp"

# Run specific test file
pytest tests/unit/test_utils.py

# Run specific test class
pytest tests/unit/test_utils.py::TestFormatTimestamp

# Run specific test function
pytest tests/unit/test_utils.py::test_format_timestamp_with_datetime_returns_formatted_string
```

### Test Markers

```bash
# Run only unit tests
pytest -m unit

# Run only integration tests  
pytest -m integration

# Skip slow tests
pytest -m "not slow"

# Run tests requiring GCP
pytest -m requires_gcp

# Run smoke tests
pytest -m smoke
```

### Coverage Options

```bash
# Basic coverage report
pytest --cov=.

# Coverage with missing lines
pytest --cov=. --cov-report=term-missing

# Coverage HTML report
pytest --cov=. --cov-report=html

# Coverage XML report (for CI/CD)
pytest --cov=. --cov-report=xml

# Fail if coverage below threshold
pytest --cov=. --cov-fail-under=60
```

### Parallel Execution

```bash
# Run tests in parallel (requires pytest-xdist)
pytest -n auto

# Run with specific number of workers
pytest -n 4
```

---

## Test Configuration

### pytest.ini

The project uses [`pytest.ini`](pytest.ini:1) for configuration:

- Test discovery patterns
- Coverage settings
- Test markers
- Warning filters
- Minimum Python version

### conftest.py

Shared fixtures in [`tests/conftest.py`](tests/conftest.py:1) include:

**File System Fixtures**:
- `temp_dir`: Temporary directory for tests
- `mock_index_dir`: Mock index directory structure
- `mock_index_files`: Complete mock index with files
- `mock_conversation_structure`: Mock conversation directory

**Data Fixtures**:
- `sample_vertex_account`: Sample account configuration
- `sample_mapping_data`: Sample index mapping
- `sample_embeddings`: Sample embedding vectors
- `sample_chunk_data`: Sample chunk data

**Mock Fixtures**:
- `mock_vertex_ai`: Mocked Vertex AI initialization
- `mock_embed_texts`: Mocked embedding function
- `mock_credentials_file`: Mock GCP credentials
- `mock_env_vars`: Environment variable mocks

---

## Writing Tests

### Test Naming Convention

```python
def test_<function>_<scenario>_<expected_result>():
    """
    Test that <function> correctly <action> when <scenario>.
    
    Given: Initial conditions
    When: Action performed
    Then: Expected outcome
    """
    # Test implementation
```

### Test Structure (AAA Pattern)

```python
def test_example():
    """Test example function behavior."""
    # Arrange: Set up test data and conditions
    input_data = {"key": "value"}
    expected = "expected_result"
    
    # Act: Execute the function under test
    result = function_under_test(input_data)
    
    # Assert: Verify the outcome
    assert result == expected
    assert len(result) > 0
```

### Using Fixtures

```python
def test_with_fixtures(temp_dir, sample_mapping_data):
    """Test using fixtures for setup."""
    # Fixtures are automatically injected
    file_path = temp_dir / "test.json"
    
    # Use fixture data
    with open(file_path, "w") as f:
        json.dump(sample_mapping_data, f)
    
    # Test assertions
    assert file_path.exists()
```

### Mocking External Dependencies

```python
from unittest.mock import patch, Mock

def test_with_mocked_api():
    """Test with mocked external API."""
    with patch('module.api_function') as mock_api:
        # Configure mock
        mock_api.return_value = {"status": "success"}
        
        # Test function that uses API
        result = function_that_calls_api()
        
        # Verify
        assert result["status"] == "success"
        mock_api.assert_called_once()
```

### Parametrized Tests

```python
import pytest

@pytest.mark.parametrize("input,expected", [
    ("hello", "HELLO"),
    ("world", "WORLD"),
    ("", ""),
])
def test_uppercase(input, expected):
    """Test uppercase conversion with multiple inputs."""
    result = input.upper()
    assert result == expected
```

---

## Test Categories

### Unit Tests

**Purpose**: Test individual functions in isolation

**Characteristics**:
- Fast (< 100ms per test)
- No external dependencies
- Use mocks for external services
- High coverage of edge cases

**Example**:
```python
@pytest.mark.unit
def test_format_timestamp():
    """Unit test for timestamp formatting."""
    dt = datetime(2024, 1, 1, 12, 0, 0)
    result = format_timestamp(dt)
    assert result == "2024-01-01 12:00:00"
```

### Integration Tests

**Purpose**: Test component interactions

**Characteristics**:
- Slower (< 5s per test)
- Test multiple components together
- May use temporary files/directories
- Verify workflows

**Example**:
```python
@pytest.mark.integration
def test_index_creation_workflow(temp_dir):
    """Integration test for complete indexing workflow."""
    # Create chunks
    chunks = create_test_chunks(temp_dir)
    
    # Generate embeddings
    embeddings = generate_embeddings(chunks)
    
    # Create index
    index = create_index(embeddings)
    
    # Verify
    assert index.is_valid()
```

---

## Common Testing Patterns

### Testing File Operations

```python
def test_file_operation(tmp_path):
    """Test file creation and reading."""
    # Create file in temp directory
    test_file = tmp_path / "test.txt"
    test_file.write_text("test content")
    
    # Test function
    result = read_file(test_file)
    
    # Verify
    assert result == "test content"
    # tmp_path is cleaned up automatically
```

### Testing with Environment Variables

```python
def test_with_env_vars(monkeypatch):
    """Test function that uses environment variables."""
    # Set test environment variables
    monkeypatch.setenv("TEST_VAR", "test_value")
    
    # Test function
    result = function_using_env_vars()
    
    # Verify
    assert result is not None
    # Environment is restored automatically
```

### Testing Logging

```python
def test_logging(caplog):
    """Test that function logs correctly."""
    with caplog.at_level(logging.INFO):
        function_that_logs()
    
    # Verify log messages
    assert "Expected message" in caplog.text
    assert caplog.records[0].levelname == "INFO"
```

### Testing Exceptions

```python
def test_exception_handling():
    """Test that function raises expected exception."""
    with pytest.raises(ValueError, match="Invalid input"):
        function_that_raises(invalid_input)
```

---

## Debugging Tests

### Run Tests in Debug Mode

```bash
# Show print statements
pytest -s

# Show local variables on failure
pytest -l

# Enter debugger on failure
pytest --pdb

# Enter debugger on first failure
pytest -x --pdb
```

### Verbose Output

```bash
# Very verbose output
pytest -vv

# Show test durations
pytest --durations=10

# Show all test outcomes
pytest -ra
```

### Running Single Test

```bash
# Run specific test with output
pytest tests/unit/test_utils.py::test_specific_function -v -s
```

---

## Continuous Integration

### GitHub Actions Example

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.10, 3.11, 3.12]
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
      
      - name: Run tests with coverage
        run: |
          pytest --cov=. --cov-report=xml
      
      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage.xml
```

---

## Coverage Analysis

### Generate Coverage Report

```bash
# Terminal report
pytest --cov=. --cov-report=term

# HTML report
pytest --cov=. --cov-report=html
open htmlcov/index.html

# XML report for CI
pytest --cov=. --cov-report=xml
```

### Coverage Thresholds

The project has coverage thresholds configured:
- **Minimum**: 60% (enforced by pytest)
- **Target**: 80% overall

```bash
# Fail if coverage below 60%
pytest --cov=. --cov-fail-under=60
```

### View Coverage by Module

```bash
# Show coverage per file
pytest --cov=. --cov-report=term-missing

# Focus on specific module
pytest --cov=diagnostics --cov-report=term
```

---

## Troubleshooting

### Common Issues

#### Import Errors

```bash
# Ensure project root is in PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Or use editable install
pip install -e .
```

#### Missing Dependencies

```bash
# Install all test dependencies
pip install -r requirements.txt

# Verify pytest is installed
pytest --version
```

#### Tests Hanging

```bash
# Use timeout to prevent hanging
pytest --timeout=300

# Or for specific test
pytest --timeout=60 tests/unit/test_specific.py
```

#### Permission Errors

```bash
# On Windows, may need to run with elevated permissions
# Or ensure temp directories are writable
pytest --basetemp=/tmp/pytest
```

### Skipping Tests

```python
# Skip a test
@pytest.mark.skip(reason="Not implemented yet")
def test_feature():
    pass

# Skip conditionally
@pytest.mark.skipif(sys.version_info < (3, 10), reason="Requires Python 3.10+")
def test_new_feature():
    pass

# Expected to fail
@pytest.mark.xfail(reason="Known bug")
def test_buggy_feature():
    pass
```

---

## Best Practices

### 1. Keep Tests Independent

```python
# Bad: Tests depend on each other
def test_step_1():
    global state
    state = "initialized"

def test_step_2():
    # Depends on test_step_1
    assert state == "initialized"

# Good: Each test is independent
def test_step_1():
    state = initialize_state()
    assert state == "initialized"

def test_step_2():
    state = initialize_state()
    assert state == "initialized"
```

### 2. Use Descriptive Names

```python
# Bad
def test_1():
    pass

# Good
def test_format_timestamp_with_valid_datetime_returns_formatted_string():
    pass
```

### 3. Test One Thing Per Test

```python
# Bad: Testing multiple things
def test_everything():
    assert function1() == expected1
    assert function2() == expected2
    assert function3() == expected3

# Good: Separate tests
def test_function1_returns_expected_value():
    assert function1() == expected1

def test_function2_returns_expected_value():
    assert function2() == expected2
```

### 4. Use Fixtures for Common Setup

```python
# Bad: Repeated setup
def test_1():
    data = create_test_data()
    # Test using data

def test_2():
    data = create_test_data()
    # Test using data

# Good: Fixture
@pytest.fixture
def test_data():
    return create_test_data()

def test_1(test_data):
    # Test using test_data

def test_2(test_data):
    # Test using test_data
```

### 5. Test Edge Cases

```python
def test_function_with_edge_cases():
    """Test function with various edge cases."""
    # Empty input
    assert function([]) == expected_empty
    
    # Single element
    assert function([1]) == expected_single
    
    # Large input
    assert function(range(10000)) == expected_large
    
    # Invalid input
    with pytest.raises(ValueError):
        function(None)
```

---

## Test Metrics

### Current Test Suite Statistics

- **Total Test Files**: 6
- **Unit Tests**: ~150 tests across 4 modules
- **Integration Tests**: ~40 tests across 2 modules
- **Total Tests**: ~190 tests
- **Test Execution Time**: 
  - Unit tests: < 5 seconds
  - Integration tests: < 30 seconds
  - Full suite: < 1 minute

### Coverage by Module

| Module | Tests | Coverage Target |
|--------|-------|----------------|
| diagnostics/utils.py | 40+ | 95% |
| diagnostics/diagnostics.py | 30+ | 90% |
| diagnostics/monitor.py | 45+ | 85% |
| diagnostics/statistics.py | 35+ | 80% |
| integration workflows | 40+ | 70% |

---

## Additional Resources

### Documentation

- [pytest Documentation](https://docs.pytest.org/)
- [pytest-cov Documentation](https://pytest-cov.readthedocs.io/)
- [TESTING_STRATEGY.md](TESTING_STRATEGY.md) - Comprehensive testing strategy
- [UNTESTABLE_CODE.md](UNTESTABLE_CODE.md) - Known testing limitations

### Internal References

- [`pytest.ini`](pytest.ini:1) - Test configuration
- [`tests/conftest.py`](tests/conftest.py:1) - Shared fixtures
- [`ARCHITECTURAL_ASSESSMENT.md`](ARCHITECTURAL_ASSESSMENT.md:1) - Code quality findings

---

## Next Steps

### For Developers

1. **Run tests before committing**:
   ```bash
   pytest -m "unit and not slow"
   ```

2. **Check coverage for your changes**:
   ```bash
   pytest --cov=your_module --cov-report=term-missing
   ```

3. **Add tests for new features**:
   - Write tests first (TDD)
   - Aim for 80%+ coverage
   - Include edge cases

### For Reviewers

1. **Verify test coverage**:
   ```bash
   pytest --cov=. --cov-report=html
   ```

2. **Run full test suite**:
   ```bash
   pytest -v
   ```

3. **Check for flaky tests**:
   ```bash
   pytest --count=10  # Run tests 10 times
   ```

---

## Support

For issues or questions about the test suite:

1. Check [TESTING_STRATEGY.md](TESTING_STRATEGY.md) for testing philosophy
2. Review [UNTESTABLE_CODE.md](UNTESTABLE_CODE.md) for known limitations
3. Examine existing tests for examples
4. Run pytest with `-h` for all options: `pytest -h`

---

*Last Updated: 2025-01-10*
*Test Suite Version: 1.0*
*Coverage Target: 80%+*