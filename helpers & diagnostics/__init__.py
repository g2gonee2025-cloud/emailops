"""
EmailOps Helpers & Diagnostics Package

A consolidated set of utilities for monitoring, testing, analyzing, and maintaining
EmailOps Vertex AI system.

Modules:
--------
- sonarqube: SonarQube integration and code quality analysis
- testing: Comprehensive testing utilities for dependencies, credentials, and runtime
- monitoring: System monitoring, indexing progress, chunk analysis, and statistics
- analysis: Code analysis, batch processing with GenAI, and remediation packages

Usage Examples:
--------------
# Run SonarQube analysis
from helpers_diagnostics import sonarqube
sonarqube.setup_and_scan()

# Test GenAI authentication
from helpers_diagnostics.testing import GenAITester
tester = GenAITester()
tester.test_all()

# Monitor indexing progress
from helpers_diagnostics.monitoring import IndexMonitor
monitor = IndexMonitor()
status = monitor.check_status()

# Run local code analysis
from helpers_diagnostics.analysis import LocalCodeAnalyzer
analyzer = LocalCodeAnalyzer()
analyzer.run_analysis()
"""

from .analysis import BatchAnalyzer, LocalCodeAnalyzer, RemediationPackageGenerator
from .monitoring import (
    ChunkAnalyzer,
    FileStatisticsAnalyzer,
    IndexMonitor,
    IndexStatus,
    ProcessInfo,
)
from .sonarqube import SonarQubeManager, SonarScanner
from .testing import (
    CredentialTester,
    DependencyVerifier,
    GenAITester,
    LLMRuntimeVerifier,
)

__version__ = "2.0.0"
__author__ = "EmailOps Team"

__all__ = [
    # Analysis
    "BatchAnalyzer",
    # Monitoring
    "ChunkAnalyzer",
    # Testing
    "CredentialTester",
    "DependencyVerifier",
    "FileStatisticsAnalyzer",
    "GenAITester",
    "IndexMonitor",
    "IndexStatus",
    "LLMRuntimeVerifier",
    "LocalCodeAnalyzer",
    "ProcessInfo",
    "RemediationPackageGenerator",
    # SonarQube
    "SonarQubeManager",
    "SonarScanner",
]
