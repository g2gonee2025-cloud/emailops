# SonarQube Analysis Setup Guide

## Issue
The SonarQube analysis failed because no SonarQube server is running at `http://localhost:9000`.

## Solution Options

### Option 1: Set Up Local SonarQube Server (Recommended for Full Analysis)

#### Using Docker (Easiest):

```bash
# Start SonarQube server
docker run -d --name sonarqube -p 9000:9000 sonarqube:latest

# Wait for server to start (takes 1-2 minutes)
# Access at: http://localhost:9000
# Default credentials: admin/admin
```

#### Generate Token:
1. Open http://localhost:9000
2. Login with admin/admin (change password on first login)
3. Go to: My Account → Security → Generate Tokens
4. Copy the token

#### Run Analysis:
```bash
# Set the token
set SONAR_TOKEN=your_generated_token

# Run analysis
python run_sonar_analysis.py
```

---

### Option 2: Use SonarCloud (Free for Public Repos)

1. Sign up at https://sonarcloud.io with GitHub/GitLab/Bitbucket
2. Create a new project
3. Get your project token
4. Update environment:

```bash
set SONAR_HOST_URL=https://sonarcloud.io
set SONAR_TOKEN=your_sonarcloud_token
set SONAR_ORGANIZATION=your_org_name
```

5. Update `sonar-project.properties`:
```properties
sonar.organization=your_org_name
sonar.projectKey=your_project_key
```

6. Run: `python run_sonar_analysis.py`

---

### Option 3: Local Static Analysis (No Server Required) ✓

Use the provided `run_local_analysis.py` script for immediate results without SonarQube server.

**Advantages:**
- No server setup required
- Instant results
- Comprehensive Python-specific checks
- Security vulnerability scanning
- Code complexity analysis
- Style violations detection

**Run:**
```bash
python run_local_analysis.py
```

This will generate a detailed HTML report with all findings.

---

## Files Created

1. **sonar-project.properties** - SonarQube project configuration
2. **run_sonar_analysis.py** - Script to download scanner and run analysis
3. **run_local_analysis.py** - Alternative local analysis (no server needed)

## Analyzed Files

The following 14 files are configured for analysis:

- emailops/config.py
- emailops/doctor.py
- emailops/email_indexer.py
- emailops/env_utils.py
- emailops/index_metadata.py
- emailops/llm_client.py
- emailops/llm_runtime.py
- emailops/processor.py
- emailops/search_and_draft.py
- emailops/summarize_email_thread.py
- emailops/text_chunker.py
- emailops/utils.py
- emailops/validators.py
- emailops_gui.py

## Next Steps

**Quick Start (No Server):**
```bash
python run_local_analysis.py
```

**Full SonarQube Analysis:**
1. Start SonarQube server (Docker or SonarCloud)
2. Set SONAR_TOKEN environment variable
3. Run: `python run_sonar_analysis.py`

## Troubleshooting

**Error: "SonarQube server can not be reached"**
- Ensure SonarQube server is running
- Check firewall settings
- Verify SONAR_HOST_URL is correct

**Error: "401 Unauthorized"**
- Generate a new token
- Set SONAR_TOKEN environment variable
- Ensure token has analysis permissions

**Error: "Project key already exists"**
- Use a different sonar.projectKey in sonar-project.properties
- Or delete the existing project from SonarQube UI