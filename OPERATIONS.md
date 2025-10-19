# Operations Runbook

## Quick Start

### Local Development
```bash
# Create environment
cp .env.example .env
# Edit .env with your actual credentials

# Setup
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Run
python -m prod.run
```

### Docker Deployment
```bash
# Build and run
docker compose up --build -d

# View logs
docker compose logs -f app

# Health check
docker compose exec app python -m prod.healthcheck
```

## Configuration

### Environment Variables

All configuration via `.env` file (see `.env.example` for template):

| Variable | Default | Description |
|----------|---------|-------------|
| `ENV` | production | Environment name |
| `LOG_LEVEL` | INFO | Logging level (DEBUG, INFO, WARNING, ERROR) |
| `LOG_FILE_ENABLED` | true | Enable file-based logging |
| `LOG_FILE_PATH` | /data/logs/app.log | Log file location |
| `STATE_DIR` | /data/state | State directory |
| `CACHE_DIR` | /data/cache | Cache directory |
| `WORKERS` | 1 | Number of worker processes |
| `SENTRY_DSN` | (empty) | Sentry error tracking DSN |

### Email Configuration
| Variable | Required | Description |
|----------|----------|-------------|
| `EMAIL_IMAP_HOST` | Yes | IMAP server hostname |
| `EMAIL_IMAP_PORT` | Yes | IMAP port (usually 993) |
| `EMAIL_SMTP_HOST` | Yes | SMTP server hostname |
| `EMAIL_SMTP_PORT` | Yes | SMTP port (usually 587) |
| `EMAIL_USERNAME` | Yes | Email account username |
| `EMAIL_PASSWORD` | Yes | Email account password |

### Storage Configuration
| Variable | Default | Description |
|----------|---------|-------------|
| `INDEX_DB_URL` | sqlite:////data/emailops.db | Database URL |

## Logging

### Log Format
Structured JSON logs with fields:
- `ts`: ISO 8601 timestamp
- `level`: Log level
- `logger`: Logger name
- `message`: Log message
- `pid`: Process ID
- `thread`: Thread name
- Custom fields: `request_id`, `component`, etc.

### Log Locations
- **stdout**: Always enabled (JSON format)
- **File**: Optional `/data/logs/app.log` (rotated daily, 7-day retention)

### Log Levels
```bash
# Development (verbose)
LOG_LEVEL=DEBUG

# Production (standard)
LOG_LEVEL=INFO

# Production (quiet)
LOG_LEVEL=WARNING
```

### Viewing Logs
```bash
# Docker logs
docker compose logs -f app

# File logs (if mounted)
tail -f data/logs/app.log | jq .

# Filter by level
docker compose logs app | grep '"level":"ERROR"'
```

## Health Checks

### Container Health
```bash
# Manual check
docker compose exec app python -m prod.healthcheck

# Expected output (healthy):
{"ok": true, "env": "production", "log_level": "INFO", "workers": 1}

# Expected output (unhealthy):
{"ok": false, "error": "...", "error_type": "..."}
```

### Healthcheck Endpoint
The Docker healthcheck runs every 30 seconds. If it fails 3 consecutive times, the container is marked unhealthy.

## Backups

### Data Directories
```bash
# Backup data directory
tar czf backup-$(date +%Y%m%d).tar.gz data/

# Restore
tar xzf backup-20251016.tar.gz
```

### Database Backup (SQLite)
```bash
# Backup database
sqlite3 data/emailops.db ".backup data/emailops-backup.db"

# Or use file copy (stop app first)
docker compose stop
cp data/emailops.db data/emailops-backup.db
docker compose start
```

## Upgrades

### Application Updates
```bash
# Pull latest code
git pull origin main

# Rebuild and restart
docker compose down
docker compose up --build -d

# Verify
docker compose logs -f app
docker compose exec app python -m prod.healthcheck
```

### Dependency Updates
```bash
# Dependabot opens weekly PRs automatically

# Manual update (if needed)
pip install --upgrade -r requirements.txt
pip freeze > requirements-frozen.txt
```

## Monitoring

### Key Metrics to Monitor
1. **Health**: Container health status
2. **Logs**: Error rate, warning rate
3. **Performance**: Response times, memory usage
4. **Resources**: CPU, memory, disk usage
5. **Network**: Connection failures, timeouts

### Alerting Rules
- Container unhealthy for >5 minutes
- Error rate >1% of requests
- Memory usage >90%
- Disk usage >85%
- Response time >5 seconds (P95)

## Troubleshooting

### Container Won't Start
```bash
# Check logs
docker compose logs app

# Check healthcheck
docker compose exec app python -m prod.healthcheck

# Verify environment
docker compose exec app env | grep -E '(ENV|LOG_LEVEL|EMAIL_)'

# Common issues:
# - Missing .env file
# - Invalid environment variables
# - Port conflicts
# - Volume permission issues
```

### Application Errors
```bash
# Enable debug logging
echo "LOG_LEVEL=DEBUG" >> .env
docker compose restart app

# Check specific component
docker compose logs app | grep '"component":"llm"'

# Verify dependencies
docker compose exec app python -c "import emailops; print(emailops.__version__)"
```

### Performance Issues
```bash
# Check resource usage
docker stats app

# Check log volume
ls -lh data/logs/

# Enable profiling (if available)
LOG_LEVEL=DEBUG docker compose restart app
```

### Database Issues
```bash
# Check database
docker compose exec app python -c "from prod.runtime_settings import RuntimeSettings; print(RuntimeSettings.from_env().index_db_url)"

# SQLite integrity check
sqlite3 data/emailops.db "PRAGMA integrity_check;"

# Vacuum/optimize
sqlite3 data/emailops.db "VACUUM;"
```

## Maintenance

### Daily
- [ ] Check container health
- [ ] Review error logs
- [ ] Monitor resource usage

### Weekly
- [ ] Review and merge Dependabot PRs
- [ ] Check disk space
- [ ] Review security advisories

### Monthly
- [ ] Backup data
- [ ] Review and rotate logs
- [ ] Performance testing
- [ ] Security scan (bandit)

### Quarterly
- [ ] Dependency audit
- [ ] Disaster recovery drill
- [ ] Capacity planning
- [ ] Security review

## Disaster Recovery

### Scenario: Data Loss
1. Stop application
2. Restore from most recent backup
3. Verify data integrity
4. Restart application
5. Monitor for issues

### Scenario: Corrupted Index
1. Stop application
2. Delete corrupted index: `rm -rf data/emailops.db`
3. Rebuild index (if tooling available)
4. Or restore from backup
5. Restart and verify

### Scenario: Container Won't Start
1. Check logs: `docker compose logs app`
2. Verify .env file exists and is valid
3. Check disk space: `df -h`
4. Check docker service: `docker info`
5. Rebuild: `docker compose build --no-cache`

## Performance Optimization

### If Running Slow
1. Check CPU/memory: `docker stats`
2. Increase workers: `WORKERS=4` in .env
3. Enable caching (if not already)
4. Profile hot paths
5. Consider scaling horizontally

### If Memory Growing
1. Check for leaks: `docker stats --no-stream`
2. Review logs for repeated errors
3. Restart periodically if needed: `docker compose restart`
4. Implement memory limits in compose file

## Security Hardening

### Additional Measures
- [ ] Enable firewall (restrict inbound/outbound)
- [ ] Implement rate limiting
- [ ] Set up intrusion detection
- [ ] Enable audit logging
- [ ] Rotate secrets regularly
- [ ] Scan containers for vulnerabilities
- [ ] Implement least-privilege access

### Security Scanning
```bash
# Scan with bandit
bandit -r emailops/

# Scan container
docker scan emailops_gui:latest

# Check dependencies
pip-audit
```

## Support Contacts

- **Technical Issues**: [Technical support email]
- **Security Issues**: [Security email]
- **Emergency**: [Emergency contact]

---

**Last Updated**: 2025-10-16  
**Next Review**: 2026-01-16  
**Maintained By**: EmailOps Team
