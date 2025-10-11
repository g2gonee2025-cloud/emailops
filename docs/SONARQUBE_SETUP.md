# SonarQube Setup Guide for EmailOps Vertex AI

This guide provides comprehensive instructions for setting up and using SonarQube for code quality analysis on the EmailOps Vertex AI project.

## Table of Contents
- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running Code Analysis](#running-code-analysis)
- [First-Time Setup](#first-time-setup)
- [Understanding Results](#understanding-results)
- [Troubleshooting](#troubleshooting)
- [Scripts Reference](#scripts-reference)

## Overview

SonarQube is an open-source platform for continuous inspection of code quality. It performs automatic reviews with static analysis of code to detect bugs, code smells, and security vulnerabilities.

### Key Features for Python Projects:
- Code coverage analysis
- Bug detection
- Security vulnerability scanning
- Code smell detection
- Technical debt evaluation
- Code duplication detection

## Prerequisites

- Docker Desktop installed and running
- WSL2 configured (for Windows users)
- Python 3.11 installed
- At least 4GB of free RAM
- Port 9000 available

## Installation

### Using Docker (Recommended)

1. **Start SonarQube:**
   ```powershell
   # Windows PowerShell
   .\sonarqube\start_sonarqube.ps1
   
   # Or in WSL/Linux
   cd sonarqube
   chmod +x start_sonarqube.sh
   ./start_sonarqube.sh
   ```

2. **Verify Installation:**
   - Open http://localhost:9000 in your browser
   - You should see the SonarQube login page

3. **Stop SonarQube:**
   ```powershell
   # Windows PowerShell
   .\sonarqube\stop_sonarqube.ps1
   
   # Or in WSL/Linux
   ./sonarqube/stop_sonarqube.sh
   ```

## Configuration

### Project Configuration

The project is configured via `sonar-project.properties` in the root directory:

```properties
# Project identification
sonar.projectKey=emailops_vertex_ai
sonar.projectName=EmailOps Vertex AI
sonar.projectVersion=1.0

# Source code settings
sonar.sources=emailops,processing,diagnostics,ui
sonar.tests=tests
sonar.python.version=3.11

# Exclusions
sonar.exclusions=**/__pycache__/**,**/*.pyc,**/tests/**,**/.git/**
```

### Quality Gates

SonarQube uses Quality Gates to ensure code meets minimum quality standards:
- **Bugs**: 0 new bugs
- **Vulnerabilities**: 0 new vulnerabilities
- **Code Smells**: < 5 new code smells
- **Coverage**: > 80% on new code
- **Duplications**: < 3% duplicated lines

## Running Code Analysis

### Method 1: Using Docker Scanner (Recommended)

```powershell
# Run analysis without authentication (first time)
.\sonarqube\run_analysis_docker.ps1

# Run analysis with token (after setup)
.\sonarqube\run_analysis_docker.ps1 -Token YOUR_TOKEN_HERE
```

### Method 2: Using Local Scanner

1. **Install SonarScanner:**
   - Download from: https://docs.sonarqube.org/latest/analyzing-source-code/scanners/sonarscanner/
   - Extract and add to PATH

2. **Run Analysis:**
   ```powershell
   .\sonarqube\run_analysis.ps1
   ```

## First-Time Setup

1. **Access SonarQube:**
   - Navigate to http://localhost:9000
   - Default credentials: `admin` / `admin`

2. **Change Password:**
   - You'll be prompted to change the admin password
   - Use a strong password and save it securely

3. **Create Authentication Token:**
   - Go to: My Account → Security → Generate Tokens
   - Name: `emailops-analysis`
   - Copy the generated token

4. **Configure Python Plugin:**
   - Administration → Marketplace
   - Search for "Python"
   - Install if not already installed

5. **Run First Analysis:**
   ```powershell
   .\sonarqube\run_analysis_docker.ps1 -Token YOUR_TOKEN_HERE
   ```

## Understanding Results

### Dashboard Overview
- **Bugs**: Programming errors that need fixing
- **Vulnerabilities**: Security issues
- **Code Smells**: Maintainability issues
- **Coverage**: Percentage of code covered by tests
- **Duplications**: Repeated code blocks

### Key Metrics for Python:
- **Cyclomatic Complexity**: Measure of code complexity
- **Cognitive Complexity**: How hard code is to understand
- **Technical Debt**: Time needed to fix all issues

### Severity Levels:
- **Blocker**: Must be fixed immediately
- **Critical**: High priority fixes
- **Major**: Should be fixed
- **Minor**: Nice to fix
- **Info**: Informational only

## Troubleshooting

### Common Issues

1. **SonarQube Won't Start:**
   ```bash
   # Check if port 9000 is in use
   netstat -an | findstr :9000
   
   # Check Docker logs
   docker logs sonarqube
   ```

2. **Analysis Fails:**
   - Ensure SonarQube is fully started (wait 30-60 seconds)
   - Check authentication token is valid
   - Verify project key matches in configuration

3. **Out of Memory:**
   - Increase Docker memory allocation
   - Reduce analysis scope in sonar-project.properties

4. **Connection Refused:**
   - Check Docker is running
   - Verify WSL2 integration
   - Ensure firewall isn't blocking port 9000

### Debug Commands

```powershell
# Check container status
docker ps

# View SonarQube logs
docker logs sonarqube

# Check database logs
docker logs sonarqube_db

# Test API endpoint
curl http://localhost:9000/api/system/status
```

## Scripts Reference

### start_sonarqube.ps1 / .sh
Starts SonarQube and PostgreSQL containers using Docker Compose.

### stop_sonarqube.ps1 / .sh
Stops and removes SonarQube containers.

### run_analysis.ps1
Runs code analysis using local SonarScanner installation.

### run_analysis_docker.ps1
Runs code analysis using Docker-based SonarScanner (no local installation required).

## Best Practices

1. **Regular Analysis:**
   - Run analysis before each commit
   - Set up CI/CD integration for automatic analysis

2. **Fix Issues Promptly:**
   - Address blockers and critical issues immediately
   - Plan refactoring for major code smells

3. **Monitor Trends:**
   - Track quality metrics over time
   - Set improvement goals

4. **Custom Rules:**
   - Configure Python-specific rules
   - Adjust quality gates based on project needs

## Integration with Development Workflow

1. **Pre-commit Hook:**
   ```bash
   # Add to .git/hooks/pre-commit
   .\sonarqube\run_analysis_docker.ps1
   ```

2. **VS Code Integration:**
   - Install SonarLint extension
   - Connect to local SonarQube server

3. **CI/CD Pipeline:**
   ```yaml
   # Example GitHub Actions
   - name: SonarQube Scan
     run: |
       docker run --network host \
         -v ${{ github.workspace }}:/usr/src \
         sonarsource/sonar-scanner-cli
   ```

## Additional Resources

- [SonarQube Documentation](https://docs.sonarqube.org/latest/)
- [Python Plugin Documentation](https://github.com/SonarSource/sonar-python)
- [SonarQube Community](https://community.sonarsource.com/)
- [Quality Gate Configuration](https://docs.sonarqube.org/latest/user-guide/quality-gates/)

---

*Last Updated: October 2024*