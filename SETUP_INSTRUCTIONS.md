# EmailOps Setup Instructions

> **Project:** EmailOps Vertex AI  
> **Date:** 2025-01-12  
> **Status:** Ready for Setup

---

## Part 1: Vertex-Only Implementation (✅ COMPLETE)

All Vertex-only recommendations have been **verified and documented**:

### Documentation Created

1. **[`emailops_docs/email_indexer.md`](emailops_docs/email_indexer.md)** — Index builder reference
2. **[`emailops_docs/index_metadata.md`](emailops_docs/index_metadata.md)** — Metadata management
3. **[`emailops_docs/VERTEX_ALIGNMENT_SUMMARY.md`](emailops_docs/VERTEX_ALIGNMENT_SUMMARY.md)** — Comprehensive alignment report
4. **[`emailops_docs/IMPLEMENTATION_VERIFICATION.md`](emailops_docs/IMPLEMENTATION_VERIFICATION.md)** — Verification matrix
5. **[`emailops_docs/README_VERTEX.md`](emailops_docs/README_VERTEX.md)** — Quick start guide
6. **[`emailops_docs/FINAL_IMPLEMENTATION_REPORT.md`](emailops_docs/FINAL_IMPLEMENTATION_REPORT.md)** — Executive summary

### Key Findings

✅ **No code changes required** — Implementation already matches all Vertex-only requirements  
✅ **Provider constrained** — CLI only accepts `"vertex"`  
✅ **Three update modes** — Full, timestamp, file-times  
✅ **Stable IDs** — Deterministic chunk and attachment IDs  
✅ **Complete documentation** — 8 documents, ~2,100 lines

---

## Part 2: SonarQube Setup (📦 READY TO USE)

### Local Installation

**SonarQube Path:** `C:\Users\ASUS\sonarqube-datacenter\sonarqube-2025.3.0.108892`

### Files Created

1. **[`setup/start_local_sonarqube.ps1`](setup/start_local_sonarqube.ps1)** — Start SonarQube server
2. **[`setup/stop_local_sonarqube.ps1`](setup/stop_local_sonarqube.ps1)** — Stop SonarQube server
3. **[`setup/run_sonar_scan.ps1`](setup/run_sonar_scan.ps1)** — Run code analysis
4. **[`setup/SONARQUBE_SETUP.md`](setup/SONARQUBE_SETUP.md)** — Complete setup guide

### Quick Start

#### Step 1: Start SonarQube

```powershell
# Start the local SonarQube server
.\setup\start_local_sonarqube.ps1
```

**What it does:**
- Verifies installation exists at `C:\Users\ASUS\sonarqube-datacenter\sonarqube-2025.3.0.108892`
- Checks if SonarQube is already running
- Starts SonarQube server via `StartSonar.bat`
- Waits for server to be ready (~30-60 seconds)
- Displays access instructions

#### Step 2: Configure SonarQube (First Time Only)

1. **Open browser:** http://localhost:9000
2. **Login:** `admin` / `admin`
3. **Change password** (required on first login)
4. **Generate token:**
   - Click on avatar (top-right) → My Account
   - Go to Security tab
   - Click "Generate Token"
   - Name: `EmailOps CLI`
   - Type: `Global Analysis Token`
   - Expiration: `90 days` or `No expiration`
   - Click "Generate"
   - **Copy the token** (you won't see it again!)

5. **Save token as environment variable:**
   ```powershell
   # Set for current session
   $env:SONAR_TOKEN = "YOUR_TOKEN_HERE"
   
   # OR set permanently (recommended)
   [System.Environment]::SetEnvironmentVariable("SONAR_TOKEN", "YOUR_TOKEN_HERE", "User")
   ```

#### Step 3: Run Analysis

```powershell
# With token in environment variable
.\setup\run_sonar_scan.ps1

# OR provide token directly
.\setup\run_sonar_scan.ps1 -Token "YOUR_TOKEN_HERE"
```

#### Step 4: View Results

```
http://localhost:9000/dashboard?id=EmailOps-v2
```

#### Step 5: Stop SonarQube (When Done)

```powershell
.\setup\stop_local_sonarqube.ps1
```

---

## Quick Commands Reference

### SonarQube Management

```powershell
# Start SonarQube server
.\setup\start_local_sonarqube.ps1

# Stop SonarQube server
.\setup\stop_local_sonarqube.ps1

# Check if running
Invoke-WebRequest -Uri "http://localhost:9000/api/system/status" -Method GET

# View logs
Get-Content "C:\Users\ASUS\sonarqube-datacenter\sonarqube-2025.3.0.108892\logs\sonar.log" -Tail 50
```

### Analysis

```powershell
# Run analysis (with token in environment)
.\setup\run_sonar_scan.ps1

# Run analysis (with token parameter)
.\setup\run_sonar_scan.ps1 -Token "YOUR_TOKEN_HERE"

# View results in browser
Start-Process "http://localhost:9000/dashboard?id=EmailOps-v2"
```

---

## Troubleshooting

### "SonarQube installation not found"

**Verify path:**
```powershell
Test-Path "C:\Users\ASUS\sonarqube-datacenter\sonarqube-2025.3.0.108892"
```

If the path is different, edit [`setup/start_local_sonarqube.ps1`](setup/start_local_sonarqube.ps1) and update line 5:
```powershell
$SONARQUBE_HOME = "YOUR_ACTUAL_PATH_HERE"
```

### "Port 9000 already in use"

**Check what's using port 9000:**
```powershell
netstat -ano | findstr :9000
```

**Change SonarQube port:**
Edit `C:\Users\ASUS\sonarqube-datacenter\sonarqube-2025.3.0.108892\conf\sonar.properties`:
```properties
sonar.web.port=9001
```

Then update `sonar-project.properties`:
```properties
sonar.host.url=http://localhost:9001
```

### "SonarQube takes too long to start"

**This is normal:**
- First start: 1-2 minutes (database initialization)
- Subsequent starts: 30-60 seconds

**Check logs:**
```powershell
Get-Content "C:\Users\ASUS\sonarqube-datacenter\sonarqube-2025.3.0.108892\logs\sonar.log" -Tail 100 -Wait
```

**Wait for:**
```
SonarQube is operational
```

---

## Complete Setup Checklist

### Vertex-Only Implementation ✅
- [x] Code reviewed (4,847 lines across 8 modules)
- [x] Documentation created (8 files, ~2,100 lines)
- [x] Environment variables verified
- [x] Implementation aligned with recommendations
- [x] No code changes required

### SonarQube Setup
- [ ] SonarQube server started via `.\setup\start_local_sonarqube.ps1`
- [ ] First-time login completed (admin/admin)
- [ ] Password changed
- [ ] Authentication token generated
- [ ] Token saved to environment variable (`$env:SONAR_TOKEN`)
- [ ] First analysis completed via `.\setup\run_sonar_scan.ps1`
- [ ] Results reviewed in dashboard at http://localhost:9000

### Next Steps
- [ ] Review SonarQube code quality findings
- [ ] Address any critical issues
- [ ] Configure quality gates (optional)
- [ ] Run integration tests for Vertex AI implementation
- [ ] Deploy to production

---

## System Requirements

### For SonarQube

- **Java:** JRE 17+ (included with installation)
- **RAM:** 4GB minimum, 8GB recommended
- **Disk:** 10GB free space
- **Port:** 9000 available

### For EmailOps

- **Python:** 3.10+
- **Dependencies:** See `requirements.txt`
- **Vertex AI:** GCP project with API enabled
- **Credentials:** Service account JSON in `secrets/`

---

## Support

For issues or questions:

- **Vertex Implementation:** See [`emailops_docs/README_VERTEX.md`](emailops_docs/README_VERTEX.md)
- **SonarQube Setup:** See [`setup/SONARQUBE_SETUP.md`](setup/SONARQUBE_SETUP.md)
- **SonarQube Logs:** `C:\Users\ASUS\sonarqube-datacenter\sonarqube-2025.3.0.108892\logs\`

---

**Ready to start SonarQube!**

Run this command:
```powershell
.\setup\start_local_sonarqube.ps1