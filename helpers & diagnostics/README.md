# EmailOps Helpers & Diagnostics

A consolidated and refactored utilities package for the EmailOps Vertex AI system.

## 🚀 Refactoring Summary

Previously this folder contained **30+ individual scripts** with overlapping functionality. These have been consolidated into **4 well-organized modules** plus supporting files.

### Before (30+ files):
- Multiple SonarQube scripts (6 files)
- Various test scripts (8 files)  
- Monitoring scripts (4 files)
- Analysis scripts (4 files)
- Example/utility scripts (8+ files)

### After (7 files):
- `sonarqube.py` - All SonarQube functionality
- `testing.py` - All testing utilities
- `monitoring.py` - All monitoring and statistics  
- `analysis.py` - Code analysis and package generation
- `__init__.py` - Package initialization
- `setup.py` - Package setup (preserved)
- `test_emailops_gui.py` - GUI testing (preserved as unique)

## 📦 Module Overview

### sonarqube.py
Consolidated from: `fix_sonar_auth.py`, `get_sonar_token.py`, `run_sonar_analysis.py`, `setup_sonar_token.py`, etc.

**Features:**
- `SonarQubeManager` - Authentication and token management
- `SonarScanner` - Scanner download and execution
- Complete setup and scan workflow
- CLI interface

**Usage:**
```python
from helpers_diagnostics.sonarqube import SonarQubeManager, SonarScanner

# Setup authentication
manager = SonarQubeManager()
token = manager.setup_token()

# Run analysis
scanner = SonarScanner()
scanner.run_analysis(token=token)
```

### testing.py
Consolidated from: `verify_dependencies.py`, `test_genai_auth.py`, `test_credentials_live.py`, `verify_llm_runtime_fixes.py`, etc.

**Features:**
- `DependencyVerifier` - Check Python dependencies
- `GenAITester` - Test GenAI/Vertex AI authentication
- `CredentialTester` - Test GCP service accounts
- `LLMRuntimeVerifier` - Verify runtime fixes

**Usage:**
```python
from helpers_diagnostics.testing import GenAITester

tester = GenAITester()
success = tester.test_all()
```

### monitoring.py
Consolidated from: `monitor.py`, `statistics.py`, `check_chunks.py`, `live_test.py`, `fix_empty_chunks.py`

**Features:**
- `IndexMonitor` - Monitor indexing progress
- `ChunkAnalyzer` - Analyze chunk files
- `FileStatisticsAnalyzer` - File statistics and processing rules
- `LiveTester` - Run live tests on conversations
- Process monitoring with psutil

**Usage:**
```python
from helpers_diagnostics.monitoring import IndexMonitor

monitor = IndexMonitor()
status = monitor.check_status()
rate = monitor.analyze_rate()
```

### analysis.py
Consolidated from: `batch_prompt.py`, `run_local_analysis.py`, `create_production_packages.py`, `generate_remediation_packages.py`

**Features:**
- `BatchAnalyzer` - Batch analysis with GenAI
- `LocalCodeAnalyzer` - Local code quality analysis
- `RemediationPackageGenerator` - Generate fix packages

**Usage:**
```python
from helpers_diagnostics.analysis import LocalCodeAnalyzer

analyzer = LocalCodeAnalyzer()
results = analyzer.run_analysis()
```

## 🔧 Command Line Usage

Each module provides a CLI interface:

```bash
# SonarQube operations
python "helpers & diagnostics/sonarqube.py" setup
python "helpers & diagnostics/sonarqube.py" scan
python "helpers & diagnostics/sonarqube.py" full

# Testing operations
python "helpers & diagnostics/testing.py" deps    # Check dependencies
python "helpers & diagnostics/testing.py" genai   # Test GenAI
python "helpers & diagnostics/testing.py" creds   # Test credentials
python "helpers & diagnostics/testing.py" runtime # Verify runtime
python "helpers & diagnostics/testing.py" all     # Run all tests

# Monitoring operations
python "helpers & diagnostics/monitoring.py" status    # Index status
python "helpers & diagnostics/monitoring.py" rate      # Indexing rate
python "helpers & diagnostics/monitoring.py" chunks    # Chunk analysis
python "helpers & diagnostics/monitoring.py" files     # File statistics
python "helpers & diagnostics/monitoring.py" processing # Processing rules
python "helpers & diagnostics/monitoring.py" live      # Live test
python "helpers & diagnostics/monitoring.py" full      # Full report

# Analysis operations
python "helpers & diagnostics/analysis.py" local     # Local analysis
python "helpers & diagnostics/analysis.py" batch --files *.py  # Batch with GenAI
python "helpers & diagnostics/analysis.py" remediate # Generate packages
```

## 📊 Benefits of Refactoring

1. **Reduced Redundancy**: Eliminated duplicate code across multiple scripts
2. **Better Organization**: Logical grouping of related functionality
3. **Easier Maintenance**: Fewer files to maintain and update
4. **Improved Imports**: Clean package structure with proper `__init__.py`
5. **Consistent CLI**: Unified command-line interfaces across modules
6. **Better Documentation**: Consolidated documentation in one place
7. **Code Reuse**: Shared utilities and common patterns

## 📝 Preserved Files

Some files were preserved as they serve unique purposes:
- `test_emailops_gui.py` - Specific GUI testing functionality
- `setup.py` - Package setup configuration
- `NEXT_STEPS_GUI_IMPLEMENTATION.md` - Implementation documentation
- `setup/` folder - Setup utilities

## 🔄 Migration Guide

If you were using the old scripts, here's how to migrate:

| Old Script | New Module | Function/Class |
|------------|------------|----------------|
| `fix_sonar_auth.py` | `sonarqube.py` | `SonarQubeManager.setup_token()` |
| `verify_dependencies.py` | `testing.py` | `DependencyVerifier.verify_all()` |
| `monitor.py` | `monitoring.py` | `IndexMonitor` |
| `statistics.py` | `monitoring.py` | `FileStatisticsAnalyzer` |
| `batch_prompt.py` | `analysis.py` | `BatchAnalyzer` |
| `run_local_analysis.py` | `analysis.py` | `LocalCodeAnalyzer` |

## 📦 Installation

No special installation needed. The package uses the same dependencies as the main EmailOps project.

Optional dependencies:
- `psutil` - For process monitoring (optional)
- `requests` - For SonarQube API calls

## 🧪 Testing

Run the comprehensive test suite:

```python
# Test all modules
from helpers_diagnostics.testing import DependencyVerifier

verifier = DependencyVerifier()
results = verifier.verify_all()
```

## 📄 License

Part of the EmailOps Vertex AI project.
