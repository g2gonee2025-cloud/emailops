# Code Quality Refactoring Guide

This guide addresses high-priority code quality issues identified by SonarQube analysis.

## Priority 1: Exception Handling (S5754)

### Problem
Exceptions are caught but not re-raised or logged, hiding errors and making debugging impossible.

**Locations**:
- `cli/src/cortex_cli/main.py:1600`
- `cli/tests/test_main_refactored.py:238`

### Pattern: WRONG ❌
```python
try:
    result = critical_operation()
except Exception:
    pass  # Silent failure - hidden error!

return result  # Returns None without user knowing why
```

### Pattern: CORRECT ✅
```python
import logging

logger = logging.getLogger(__name__)

try:
    result = critical_operation()
except ValueError as e:
    # Specific exception handling
    logger.error(f"Invalid input in critical_operation: {e}", exc_info=True)
    raise  # Re-raise after logging
except KeyError as e:
    # Domain-specific exception
    from cortex.common.exceptions import ConfigurationError
    logger.error(f"Missing configuration key: {e}", exc_info=True)
    raise ConfigurationError(f"Missing config: {e}") from e
except Exception as e:
    # Catch-all with logging
    logger.exception(f"Unexpected error in critical_operation: {e}")
    raise  # Always re-raise unexpected exceptions

return result  # Only reached if no exception
```

### Exception Handling Guidelines

1. **Always log before re-raising**:
   ```python
   try:
       # operation
   except SpecificException as e:
       logger.error(f"Context: {e}", exc_info=True)  # exc_info includes traceback
       raise
   ```

2. **Use specific exception types**:
   ```python
   # WRONG: Too broad
   except Exception:
       pass
   
   # CORRECT: Specific
   except (ValueError, TypeError, KeyError) as e:
       logger.error(f"Data error: {e}")
       raise
   ```

3. **Provide context in re-raised exceptions**:
   ```python
   try:
       db_result = session.query(User).filter_by(id=user_id).one()
   except NoResultFound as e:
       logger.error(f"User not found: id={user_id}")
       raise UserNotFoundError(f"No user with id {user_id}") from e
   ```

4. **Use exception chaining** for context preservation:
   ```python
   try:
       parse_json(config_str)
   except json.JSONDecodeError as e:
       raise ConfigurationError(f"Invalid JSON in config") from e
   ```

## Priority 2: High Cognitive Complexity

### Problem
108 functions exceed cognitive complexity threshold (>15). These are difficult to understand, test, and maintain.

**Top offenders**:
- `kube-state-metrics/pkg/customresourcestate/registry_factory.go:582` (complexity: 108)
- `backend/src/cortex/intelligence/graph.py:386` (complexity: 77)
- `backend/src/cortex/rag_api/routes_ingest.py:519` (complexity: 51)
- `kube-state-metrics/pkg/app/server.go:90` (complexity: 42)

### Refactoring Pattern: Extract Method

**BEFORE** (Complexity: 45)
```python
def process_documents(documents, config, filters, cache):
    results = []
    for doc in documents:
        if doc.status == "active":
            if filters.get('by_type') and doc.type != filters['by_type']:
                continue
            if filters.get('by_date'):
                if doc.date < filters['by_date']:
                    continue
            
            cached = cache.get(doc.id)
            if cached:
                results.append(cached)
            else:
                processed = process_single_document(doc, config)
                if processed:
                    cache.set(doc.id, processed)
                    results.append(processed)
                else:
                    logger.warning(f"Failed to process {doc.id}")
        else:
            logger.debug(f"Skipping inactive document {doc.id}")
    
    return results
```

**AFTER** (Complexity: 3 per function)
```python
def _matches_filters(doc: Document, filters: Dict) -> bool:
    """Check if document matches specified filters."""
    if filters.get('by_type') and doc.type != filters['by_type']:
        return False
    if filters.get('by_date') and doc.date < filters['by_date']:
        return False
    return True


def _get_processed_doc(doc: Document, config: Config, cache: Cache) -> Optional[Document]:
    """Get processed document from cache or process it."""
    cached = cache.get(doc.id)
    if cached:
        return cached
    
    processed = process_single_document(doc, config)
    if not processed:
        logger.warning(f"Failed to process {doc.id}")
        return None
    
    cache.set(doc.id, processed)
    return processed


def process_documents(
    documents: List[Document],
    config: Config,
    filters: Dict,
    cache: Cache
) -> List[Document]:
    """Process filtered documents with caching."""
    results = []
    
    for doc in documents:
        if doc.status != "active":
            logger.debug(f"Skipping inactive document {doc.id}")
            continue
        
        if not _matches_filters(doc, filters):
            continue
        
        processed = _get_processed_doc(doc, config, cache)
        if processed:
            results.append(processed)
    
    return results
```

### Refactoring Techniques

1. **Extract Method**: Break large functions into smaller, focused ones
   ```python
   # Before: 45 lines, complexity 20
   def validate_and_process(data):
       # validation logic (15 lines)
       # processing logic (15 lines)
       # error handling (15 lines)
   
   # After: 3 functions, complexity 3-5 each
   def _validate(data) -> bool:
   def _process(data) -> Result:
   def validate_and_process(data) -> Result:
   ```

2. **Early Return**: Reduce nesting depth
   ```python
   # Before: 4 levels of nesting
   if condition1:
       if condition2:
           if condition3:
               if condition4:
                   do_something()
   
   # After: 0 levels of nesting
   if not condition1:
       return
   if not condition2:
       return
   if not condition3:
       return
   if not condition4:
       return
   do_something()
   ```

3. **Guard Clauses**: Inverse conditions at start
   ```python
   # Before: 25 lines with nested ifs
   def process_user(user):
       if user:
           if user.active:
               if user.verified:
                   # 20 lines of actual logic
   
   # After: Explicit preconditions
   def process_user(user: User) -> Result:
       if not user:
           raise ValueError("User required")
       if not user.active:
           raise UserInactiveError()
       if not user.verified:
           raise UserNotVerifiedError()
       # 20 lines of actual logic
   ```

## Priority 3: Floating Point Comparisons

### Problem
40+ instances of using `==` for float comparisons, causing flaky tests.

**Locations**: Throughout `backend/tests/orchestration/test_nodes.py`

### Pattern: WRONG ❌
```python
import unittest

class TestCalculations(unittest.TestCase):
    def test_embedding_similarity(self):
        result = calculate_similarity(vector1, vector2)
        self.assertEqual(result, 0.987654321)  # Flaky! Precision issues
```

### Pattern: CORRECT ✅
```python
import pytest

def test_embedding_similarity():
    result = calculate_similarity(vector1, vector2)
    assert result == pytest.approx(0.987654321, rel=1e-6)  # ✓ Accurate

# Or using math.isclose
import math
def test_embedding_similarity():
    result = calculate_similarity(vector1, vector2)
    assert math.isclose(result, 0.987654321, rel_tol=1e-6)  # ✓ Accurate
```

### Refactoring Script

```python
import re

def fix_float_comparisons(file_path: str) -> None:
    """Replace float equality with pytest.approx."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Find: self.assertEqual(value, float)
    pattern = r'self\.assertEqual\(([^,]+),\s*([0-9.]+)\)'
    replacement = r'assert \1 == pytest.approx(\2, rel=1e-6)'
    
    fixed = re.sub(pattern, replacement, content)
    
    # Add pytest import if not present
    if 'import pytest' not in fixed:
        fixed = 'import pytest\n' + fixed
    
    with open(file_path, 'w') as f:
        f.write(fixed)
    
    print(f"Fixed float comparisons in {file_path}")
```

## Priority 4: Synchronous I/O in Async Functions

### Problem
Synchronous file operations in async functions block the event loop.

**Locations**:
- `retry_failed_sessions.py:191`
- `summarize_jules_outcomes.py:80`
- `bulk_jules_review.py:276`

### Pattern: WRONG ❌
```python
async def process_files(file_list):
    results = []
    for file_path in file_list:
        with open(file_path, 'r') as f:  # BLOCKS event loop!
            content = f.read()  # Synchronous I/O
        
        processed = await expensive_async_operation(content)
        results.append(processed)
    
    return results
```

### Pattern: CORRECT ✅
```python
import aiofiles
import asyncio

async def process_files(file_list):
    """Process multiple files concurrently without blocking."""
    results = []
    
    async def process_single_file(file_path: str):
        # Non-blocking file I/O
        async with aiofiles.open(file_path, 'r') as f:
            content = await f.read()
        
        processed = await expensive_async_operation(content)
        return processed
    
    # Process all files concurrently
    results = await asyncio.gather(
        *[process_single_file(f) for f in file_list]
    )
    return results
```

### Installation
```bash
pip install aiofiles
```

## Priority 5: String Literal Deduplication

### Problem
90+ hardcoded string literals scattered throughout codebase.

**Example**: `"cortex.orchestration.nodes"` appears 19 times

### Pattern: WRONG ❌
```python
# module1.py
module = importlib.import_module("cortex.orchestration.nodes")

# module2.py
logger = logging.getLogger("cortex.orchestration.nodes")

# module3.py
config = settings.get("cortex.orchestration.nodes")
```

### Pattern: CORRECT ✅
```python
# cortex/constants.py
MODULE_ORCHESTRATION_NODES = "cortex.orchestration.nodes"
LOGGER_ORCHESTRATION = "cortex.orchestration"

# module1.py
from cortex.constants import MODULE_ORCHESTRATION_NODES
module = importlib.import_module(MODULE_ORCHESTRATION_NODES)

# module2.py
from cortex.constants import LOGGER_ORCHESTRATION
logger = logging.getLogger(LOGGER_ORCHESTRATION)

# module3.py
from cortex.constants import MODULE_ORCHESTRATION_NODES
config = settings.get(MODULE_ORCHESTRATION_NODES)
```

### Benefits
- Single source of truth
- Prevents typos
- Easy refactoring (search and replace)
- Better IDE autocompletion

## Implementation Checklist

- [ ] Review all try-except blocks and add logging + re-raise
- [ ] Refactor functions with complexity > 30 using Extract Method pattern
- [ ] Replace all `==` float comparisons with `pytest.approx()`
- [ ] Replace `open()` with `aiofiles.open()` in async functions
- [ ] Extract string literals to constants module
- [ ] Replace `datetime.utcnow()` with `datetime.now(timezone.utc)`
- [ ] Run SonarQube analysis and verify improvements
- [ ] Update unit tests with proper exception handling tests

## Verification Commands

```bash
# Check exception handling
grep -r "except.*:$" --include="*.py" . | grep -v "#" | grep -v "raise"

# Find float comparisons
grep -r "assertEqual.*\.[0-9]\+" --include="*.py" .

# Find synchronous I/O in async functions
grep -B5 "open(" --include="*.py" . | grep -B5 "async def"

# Find string literal duplication
grep -r 'cortex\.orchestration\.nodes' --include="*.py" .

# Run SonarQube analysis
sonar-scanner \
  -Dsonar.projectKey=emailops \
  -Dsonar.sources=backend,cli \
  -Dsonar.host.url=https://sonarqube.example.com
```

## References

- [PEP 8 Exception Handling](https://pep8.org/#exception-handling)
- [Python Exception Best Practices](https://docs.python.org/3/library/exceptions.html)
- [Cognitive Complexity in Software Engineering](https://www.sonarsource.com/docs/plugins/java/cognitive-complexity/)
- [aiofiles Documentation](https://github.com/Tinche/aiofiles)
- [pytest.approx Documentation](https://docs.pytest.org/en/stable/reference.html#pytest.approx)
