# SonarQube Quick Reference

## Quick Start

1. **Start SonarQube:**
   ```powershell
   .\sonarqube\start_sonarqube.ps1
   ```

2. **Access Web Interface:**
   - URL: http://localhost:9000
   - Default: admin/admin (change on first login)

3. **Run Analysis:**
   ```powershell
   .\sonarqube\run_analysis_docker.ps1
   ```

4. **Stop SonarQube:**
   ```powershell
   .\sonarqube\stop_sonarqube.ps1
   ```

## Files Created

- `docker-compose.yml` - Docker configuration
- `start_sonarqube.ps1/.sh` - Start scripts
- `stop_sonarqube.ps1/.sh` - Stop scripts
- `run_analysis.ps1` - Analysis with local scanner
- `run_analysis_docker.ps1` - Analysis with Docker
- `../sonar-project.properties` - Project configuration
- `../docs/SONARQUBE_SETUP.md` - Full documentation

## Key Commands

```bash
# Check status
docker ps

# View logs
docker logs sonarqube

# Test API
curl http://localhost:9000/api/system/status
```

## Project Configuration

- **Project Key:** emailops_vertex_ai
- **Sources:** emailops, processing, diagnostics, ui
- **Tests:** tests/
- **Python Version:** 3.11

See `docs/SONARQUBE_SETUP.md` for complete documentation.