# Production Deployment Guide for EmailOps

This guide covers deploying EmailOps in production with comprehensive hardening, monitoring, and operational readiness.

## 🚀 Quick Start

```bash
# 1. Clone and setup
git clone <repository>
cd emailops_vertex_ai

# 2. Configure environment
cp .env.example .env
# Edit .env with your actual credentials

# 3. Build and run
docker compose up --build -d

# 4. Verify
docker compose logs -f app
docker compose exec app python -m prod.healthcheck
```

## 📦 What's Included

### Production Runtime (`prod/`)
- **`_logging_setup.py`**: Structured JSON logging
- **`runtime_settings.py`**: Environment-driven configuration
- **`healthcheck.py`**: Container health verification
- **`run.py`**: Safe production entrypoint

### Quality Assurance
- **Linting**: Ruff for code quality
- **Type Checking**: mypy for type safety
- **Security Scanning**: Bandit for vulnerability detection
- **Pre-commit Hooks**: Automatic quality checks

### Containerization
- **Dockerfile**: Multi-stage build, non-root user, health checks
- **docker-compose.yml**: Local orchestration
- **CI/CD**: GitHub Actions workflow

### Documentation
- **SECURITY.md**: Security policy and vulnerability reporting
- **OPERATIONS.md**: Comprehensive operations runbook
- **This file**: Production deployment guide

## 🏗️ Architecture

```
EmailOps Production Stack
├── Container (python:3.11-slim)
│   ├── App Code (emailops/)
│   ├── Production Runtime (prod/)
│   ├── Data Volumes (/data)
│   └── Non-root User (appuser)
├── Logging
│   ├── JSON to stdout → Log aggregation
│   └── File logs → Local persistence
├── Health Checks
│   ├── Runtime validation
│   └── Orchestration integration
└── CI/CD Pipeline
    ├── Linting & Type Checking
    ├── Security Scanning
    └── Automated Testing
```

## 🔧 Configuration Management

### Configuration Hierarchy
1. **Environment Variables** (highest priority)
2. `.env` file
3. Code defaults (lowest priority)

### Required Configuration

Minimum `.env` for production:
```bash
ENV=production
LOG_LEVEL=INFO

# Email credentials (REQUIRED)
EMAIL_IMAP_HOST=imap.yourdomain.com
EMAIL_IMAP_PORT=993
EMAIL_SMTP_HOST=smtp.yourdomain.com
EMAIL_SMTP_PORT=587
EMAIL_USERNAME=your@email.com
EMAIL_PASSWORD=YourSecurePassword123!

# Storage
INDEX_DB_URL=sqlite:////data/emailops.db
STATE_DIR=/data/state
CACHE_DIR=/data/cache
```

### Optional Configuration

```bash
# Observability
SENTRY_DSN=https://your-sentry-dsn@sentry.io/project

# Performance
WORKERS=4  # Increase for parallel processing

# Logging
LOG_FILE_ENABLED=true
LOG_FILE_PATH=/data/logs/app.log
```

## 📊 Monitoring & Observability

### Health Checks

**Container Level**:
```bash
docker compose exec app python -m prod.healthcheck
```

**Expected Output**:
```json
{"ok": true, "env": "production", "log_level": "INFO", "workers": 1}
```

### Log Monitoring

**View Real-time Logs**:
```bash
# All logs
docker compose logs -f app

# With JSON parsing
docker compose logs -f app | jq .

# Filter by level
docker compose logs app | jq 'select(.level=="ERROR")'

# Filter by component
docker compose logs app | jq 'select(.component=="llm")'
```

### Metrics to Track

| Metric | Target | Alert Threshold |
|--------|--------|-----------------|
| Container Health | Healthy | Unhealthy >5 min |
| Error Rate | <0.1% | >1% |
| Memory Usage | <70% | >90% |
| CPU Usage | <60% | >85% |
| Response Time (P95) | <2s | >5s |
| Log Errors/min | <10 | >100 |

## 🔒 Security

### Secrets Management

**Never Commit**:
- `.env` file
- Service account JSONs
- API keys or passwords
- Database credentials

**Always Use**:
- Environment variables
- Secret management services (AWS Secrets Manager, HashiCorp Vault)
- `.env.example` with safe sample values

### Security Checklist

Before production deployment:
- [ ] All secrets in environment variables
- [ ] `.env` file is gitignored
- [ ] Container runs as non-root (appuser)
- [ ] No hardcoded credentials in code
- [ ] Bandit security scan passing
- [ ] Dependencies reviewed for CVEs
- [ ] Network egress restricted (if applicable)
- [ ] TLS/SSL enabled for external connections
- [ ] Rate limiting configured
- [ ] Audit logging enabled

### Security Scanning

```bash
# Scan codebase
bandit -r emailops/

# Scan container
docker scan emailops_gui:latest

# Check dependencies
pip-audit  # if installed
```

## 🚢 Deployment Strategies

### Development
```bash
ENV=development LOG_LEVEL=DEBUG docker compose up
```

### Staging
```bash
ENV=staging LOG_LEVEL=INFO docker compose up -d
```

### Production

**Blue-Green Deployment**:
```bash
# Deploy green
docker compose -f docker-compose.prod.yml up -d green

# Test green
curl http://green:9999/health

# Switch traffic
# (configure load balancer/reverse proxy)

# Decommission blue
docker compose -f docker-compose.prod.yml stop blue
```

**Rolling Update**:
```bash
# Build new image
docker compose build

# Pull and restart (minimizes downtime)
docker compose up -d --no-deps --build app
```

## 🔥 Troubleshooting

### Common Issues

**Issue: Container Exits Immediately**
```bash
# Check logs
docker compose logs app

# Common causes:
# - Invalid .env configuration
# - Missing required environment variables
# - Python import errors
# - Database connection failures
```

**Issue: Healthcheck Failing**
```bash
# Run healthcheck manually
docker compose exec app python -m prod.healthcheck

# Check configuration validation
docker compose exec app python -c "from prod.runtime_settings import RuntimeSettings; RuntimeSettings.from_env().validate()"
```

**Issue: High Memory Usage**
```bash
# Check memory
docker stats app

# Review logs for memory-intensive operations
docker compose logs app | jq 'select(.message | contains("memory"))'

# Set memory limit
# In docker-compose.yml, add:
# deploy:
#   resources:
#     limits:
#       memory: 2G
```

**Issue: Slow Performance**
```bash
# Increase workers
echo "WORKERS=4" >> .env
docker compose restart app

# Enable DEBUG logging
echo "LOG_LEVEL=DEBUG" >> .env
docker compose restart app

# Check for bottlenecks in logs
docker compose logs app | jq 'select(.level=="WARNING" or .level=="ERROR")'
```

## 🧪 Testing

### Unit Tests
```bash
pytest tests/unit/ -v
```

### Integration Tests
```bash
pytest tests/integration/ -v
```

### End-to-End Tests
```bash
# With running container
docker compose exec app pytest tests/e2e/ -v
```

### Load Testing
```bash
# (If you have load testing tools)
locust -f tests/load/locustfile.py --host=http://localhost:9999
```

## 📈 Performance Tuning

### Optimize for Throughput
```bash
# Increase workers
WORKERS=8

# Increase connection pool sizes (if applicable)
# Tune garbage collection (if needed)
```

### Optimize for Latency
```bash
# Reduce LOG_LEVEL
LOG_LEVEL=WARNING

# Disable file logging
LOG_FILE_ENABLED=false

# Optimize database queries
```

### Resource Limits

In `docker-compose.yml`:
```yaml
deploy:
  resources:
    limits:
      cpus: '2.0'
      memory: 4G
    reservations:
      cpus: '0.5'
      memory: 1G
```

## 🔄 Backup & Recovery

### Automated Backups

Create a backup script (`scripts/backup.sh`):
```bash
#!/bin/bash
BACKUP_DIR="/backups"
DATE=$(date +%Y%m%d_%H%M%S)

# Backup data
tar czf "$BACKUP_DIR/data_$DATE.tar.gz" data/

# Backup config
cp .env "$BACKUP_DIR/env_$DATE"

# Rotate old backups (keep last 7 days)
find "$BACKUP_DIR" -name "data_*.tar.gz" -mtime +7 -delete

echo "Backup completed: $BACKUP_DIR/data_$DATE.tar.gz"
```

Run via cron:
```bash
0 2 * * * /path/to/scripts/backup.sh
```

### Recovery Procedure

```bash
# 1. Stop application
docker compose down

# 2. Restore data
tar xzf backups/data_20251016_020000.tar.gz

# 3. Verify data integrity
ls -lh data/

# 4. Restart
docker compose up -d

# 5. Verify health
docker compose exec app python -m prod.healthcheck
```

## 📞 Support & Escalation

### Support Tiers

**Tier 1**: Operational issues (restarts, config changes)
- Contact: DevOps team
- SLA: 4 hours

**Tier 2**: Application errors (bugs, performance issues)
- Contact: Engineering team
- SLA: 1 business day

**Tier 3**: Security incidents
- Contact: Security team immediately
- SLA: Immediate response

### Escalation Path

1. Check OPERATIONS.md for common solutions
2. Review recent logs for errors
3. Contact Tier 1 support
4. Escalate to Tier 2 if needed
5. Emergency: Contact Tier 3 for security issues

## 🎯 Success Criteria

Production deployment is successful when:

✅ Container starts and stays healthy
✅ Healthcheck passes consistently
✅ No errors in logs (first hour)
✅ Performance meets SLAs
✅ Security scan clean
✅ Backups configured and tested
✅ Monitoring/alerting active
✅ Team trained on operations
✅ Runbooks accessible
✅ Escalation path clear

## 📚 Additional Resources

- [SECURITY.md](SECURITY.md) - Security policy
- [OPERATIONS.md](OPERATIONS.md) - Detailed operations guide
- [remediation_packages/](remediation_packages/) - Code remediation packages
- [emailops_docs/](emailops_docs/) - Technical documentation

---

**Production Ready**: Yes (with configuration)  
**Deployment Complexity**: Low  
**Operational Overhead**: Low  
**Recommended Team Size**: 1-2 engineers

**Questions?** Review OPERATIONS.md or contact the team.
