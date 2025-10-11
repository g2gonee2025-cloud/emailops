# EmailOps Testing Strategy

## Overview

This document outlines the comprehensive testing strategy for the EmailOps project, focusing on achieving robust test coverage and maintaining code quality.

## Testing Philosophy

### Goals
- **Target Coverage**: 80%+ overall, with minimum 60% baseline
- **Quality Over Quantity**: Focus on meaningful tests that catch real bugs
- **Fast Feedback**: Unit tests run in < 5 seconds, full suite in < 2 minutes
- **Maintainability**: Tests should be clear, focused, and easy to update

### Principles
1. **Test Behavior, Not Implementation**: Focus on what code does, not how
2. **Isolation**: Unit tests should not depend on external services or file systems
3. **Repeatability**: Tests must produce consistent results
4. **Clarity**: Test names and assertions should be self-documenting

## Test Pyramid

```
       /\
      /  \     E2E (5%)
     /----\    Integration (15%)
    /------\   Unit (80%)
   /________\
```

### Distribution Target
- **Unit Tests**: 80% - Fast, isolated, test individual functions/classes
- **Integration Tests**: 15% - Test component interactions
- **End-to-End Tests**: 5% - Full workflow validation

## Testing Infrastructure

### Tools
- **pytest**: Test framework
- **pytest-cov**: Coverage reporting
- **pytest-mock**: Mocking utilities
- **pytest-timeout**: Prevent hanging tests
- **pytest-xdist**: Parallel test execution

### Configuration
- `pytest.ini`: Test discovery, markers, coverage settings
- `tests/conftest.py`: Shared fixtures and utilities
- Coverage threshold: 60% minimum, 80% target

## Test Organization

### Directory Structure
```
tests/
├── conftest.py           # Shared fixtures
├── unit/                 # Unit tests
│   ├── test_utils.py
│   ├── test_diagnostics.py
│   ├── test_monitor.py
│   ├── test_statistics.py
│   └── test_processor_core.py
├── integration/          # Integration tests
│   ├── test_index_workflow.py
│   └── test_embedding_pipeline.py
└── test_data/            # Test fixtures and data
```

### Naming Conventions
- Test files: `test_*.py` or `*_test.py`
- Test classes: `Test*` (optional, use for grouping)
- Test functions: `test_<function>_<scenario>_<expected_result>`

Example:
```python
def test_format_timestamp_with_valid_datetime_returns_formatted_string():
    """Test that format_timestamp correctly formats a datetime object."""
    ...
```

## Mocking Strategy

### External Dependencies to Mock

#### GCP/Vertex AI
- **vertexai.init()**: Always mock in unit tests
- **embed_texts()**: Return normalized random vectors
- **Credentials**: Use mock credential files in temp directories

#### File System
- Use `tmp_path` fixture for file operations
- Mock `Path.exists()`, `Path.read_text()` for isolation
- Create minimal test data structures

#### Streamlit
- Mock all streamlit UI components
- Focus on testing business logic, not UI rendering

### Mock Patterns

#### Pattern 1: Simple Function Mock
```python
@patch('module.function_name')
def test_something(mock_function):
    mock_function.return_value = expected_value
    result = code_under_test()
    assert result == expected
```

#### Pattern 2: Context Manager Mock
```python
with patch('module.Class') as MockClass:
    instance = MockClass.return_value
    instance.method.return_value = value
    result = code_under_test()
```

#### Pattern 3: Fixture-Based Mock
```python
@pytest.fixture
def mock_service():
    with patch('service.ServiceClass') as mock:
        yield mock

def test_with_fixture(mock_service):
    # Test using the fixture
    ...
```

## Coverage Requirements by Module

### High Priority (80%+ coverage target)

#### diagnostics/utils.py
- All utility functions must be tested
- Focus on edge cases: empty inputs, None values, invalid paths

#### diagnostics/diagnostics.py
- Account validation logic
- Index verification functions
- Error handling paths

#### diagnostics/monitor.py
- IndexMonitor class methods
- Progress calculation
- Time estimation

#### diagnostics/statistics.py
- File counting and statistics
- Chunk analysis
- Progress monitoring

### Medium Priority (60%+ coverage target)

#### processing/processor.py
- **Note**: Currently wrapped in string literal (non-functional)
- Test individual helper functions when extracted
- Focus on chunking and embedding logic once fixed

#### setup/enable_vertex_apis.py
- API enabling logic
- Project validation
- Error scenarios

### Lower Priority (40%+ coverage target)

#### ui/emailops_ui.py
- Business logic only (not Streamlit UI)
- State management
- Data validation

## Test Categories and Markers

### Markers
```python
@pytest.mark.unit          # Fast, isolated unit tests
@pytest.mark.integration   # Integration tests
@pytest.mark.slow          # Tests taking > 1 second
@pytest.mark.requires_gcp  # Needs GCP credentials
@pytest.mark.requires_files # Needs file system access
@pytest.mark.smoke         # Smoke tests for CI
```

### Running Tests by Category
```bash
# Run only unit tests
pytest -m unit

# Run only fast tests
pytest -m "not slow"

# Run smoke tests
pytest -m smoke

# Run with coverage
pytest --cov=. --cov-report=html
```

## Fixture Design Patterns

### Hierarchical Fixtures
```python
@pytest.fixture
def base_config():
    return {"setting": "value"}

@pytest.fixture
def extended_config(base_config):
    config = base_config.copy()
    config["extra"] = "data"
    return config
```

### Parameterized Fixtures
```python
@pytest.fixture(params=[
    {"input": "a", "expected": 1},
    {"input": "b", "expected": 2},
])
def test_case(request):
    return request.param
```

### Cleanup Fixtures
```python
@pytest.fixture
def resource():
    obj = create_resource()
    yield obj
    obj.cleanup()  # Always runs, even if test fails
```

## Test Implementation Guidelines

### Unit Test Template
```python
def test_function_name_with_scenario_returns_expected():
    """
    Test that function_name correctly handles scenario.
    
    Given: Initial conditions
    When: Action performed
    Then: Expected outcome
    """
    # Arrange
    input_data = setup_test_data()
    expected = calculate_expected()
    
    # Act
    result = function_under_test(input_data)
    
    # Assert
    assert result == expected
    assert_additional_conditions(result)
```

### Edge Cases to Test

#### For All Functions
- Empty inputs ([], "", None)
- Single element inputs
- Large inputs (performance)
- Invalid types
- Boundary values

#### For File Operations
- File not found
- Permission errors
- Invalid file format
- Corrupted data
- Empty files

#### For Numeric Operations
- Zero values
- Negative values
- Very large numbers
- NaN and infinity
- Division by zero

#### For String Operations
- Empty strings
- Unicode characters
- Very long strings
- Special characters
- Whitespace handling

## Integration Test Guidelines

### Workflow Testing
1. **Setup**: Create minimal required state
2. **Execute**: Run the workflow
3. **Verify**: Check multiple checkpoints
4. **Cleanup**: Tear down state

### Example Integration Test
```python
def test_indexing_workflow_creates_valid_index(temp_dir):
    """
    Test complete indexing workflow from chunks to index.
    
    This integration test verifies:
    1. Chunks are read correctly
    2. Embeddings are generated
    3. Index files are created
    4. Index is queryable
    """
    # Setup
    create_test_chunks(temp_dir)
    
    # Execute
    processor = UnifiedProcessor(temp_dir)
    processor.create_embeddings()
    processor.repair_index()
    
    # Verify
    assert (temp_dir / "_index" / "mapping.json").exists()
    assert (temp_dir / "_index" / "embeddings.npy").exists()
    
    # Validate structure
    mapping = load_mapping(temp_dir / "_index")
    assert len(mapping) > 0
    assert all("id" in entry for entry in mapping)
```

## CI/CD Integration Plan

### GitHub Actions Workflow
```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - run: pip install -r requirements.txt
      - run: pytest --cov=. --cov-report=xml
      - uses: codecov/codecov-action@v3
```

### Pre-commit Hooks
```yaml
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: pytest-check
        name: pytest-check
        entry: pytest -m "unit and not slow"
        language: system
        pass_filenames: false
        always_run: true
```

## Coverage Tracking

### Target Metrics
- **Overall**: 80%+
- **Critical modules**: 90%+ (utils, diagnostics core)
- **Integration modules**: 60%+ (UI, processing)

### Coverage Reports
```bash
# Generate HTML report
pytest --cov=. --cov-report=html

# View in browser
open htmlcov/index.html

# Check coverage threshold
pytest --cov=. --cov-fail-under=60
```

### Coverage Exclusions
- `if __name__ == "__main__":`
- `TYPE_CHECKING` blocks
- Abstract methods
- Deprecation warnings

## Known Testing Challenges

### 1. processor.py String Wrapper
**Issue**: Entire file wrapped in string literal
**Status**: Untestable until fixed
**Solution**: Code mode must extract and fix the code first

### 2. Streamlit UI Components
**Issue**: Heavy coupling with Streamlit state
**Status**: Low priority for testing
**Solution**: Test business logic only, mock UI components

### 3. GCP API Dependencies
**Issue**: Real API calls are slow and require credentials
**Status**: Must mock for unit tests
**Solution**: Comprehensive mocking strategy in conftest.py

### 4. Large File Operations
**Issue**: Testing with full-scale data is slow
**Status**: Use minimal test data
**Solution**: Synthetic small datasets in test fixtures

## Test Data Management

### Test Data Location
- `tests/test_data/`: Committed test fixtures
- `tmp_path`: Generated temporary test data

### Test Data Guidelines
1. **Minimal**: Use smallest data that tests the behavior
2. **Representative**: Data should represent real scenarios
3. **Deterministic**: Same inputs produce same outputs
4. **Documented**: Explain what data represents

## Maintenance and Evolution

### Adding New Tests
1. Identify the function/feature to test
2. Choose appropriate test category (unit/integration)
3. Add test markers
4. Follow naming conventions
5. Document test purpose
6. Update coverage tracking

### Refactoring Tests
1. Keep tests passing during refactoring
2. Update tests to match new interfaces
3. Maintain test independence
4. Preserve test coverage

### Deprecating Tests
1. Mark as deprecated with comment
2. Keep passing for transition period
3. Remove after migration complete

## Success Metrics

### Quantitative
- Coverage: 60%+ baseline, 80%+ target
- Test count: 50+ unit, 10+ integration
- Test speed: < 5s unit, < 2min full suite
- Failure rate: < 1% flaky tests

### Qualitative
- Tests catch real bugs before production
- Tests are easy to understand and maintain
- Tests provide confidence for refactoring
- Tests document expected behavior

## Resources and References

### Documentation
- [pytest documentation](https://docs.pytest.org/)
- [pytest-cov](https://pytest-cov.readthedocs.io/)
- [Python testing best practices](https://docs.python-guide.org/writing/tests/)

### Internal
- `tests/conftest.py`: Fixture definitions
- `pytest.ini`: Configuration
- `ARCHITECTURAL_ASSESSMENT.md`: Code quality findings

## Conclusion

This testing strategy provides a comprehensive framework for achieving high-quality, maintainable test coverage for the EmailOps project. By following these guidelines, we ensure code reliability, facilitate refactoring, and maintain confidence in the system's behavior.

---

*Last Updated: 2025-01-10*
*Coverage Target: 80%*
*Current Coverage: TBD (after initial test implementation)*