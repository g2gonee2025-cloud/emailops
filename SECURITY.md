# Security Policy

## Supported Versions

The `main` branch is actively supported. Pin dependencies and use Dependabot PRs for updates.

| Version | Supported          |
| ------- | ------------------ |
| main    | :white_check_mark: |
| older   | :x:                |

## Reporting a Vulnerability

**Please do not file public issues for undisclosed vulnerabilities.**

Instead:
1. Email the maintainers directly
2. Open a private security advisory on GitHub
3. Provide details: affected component, attack vector, proposed fix

We aim to respond within 48 hours and provide fixes within 7 days for critical issues.

## Secrets Management

### Never Commit
- API keys, passwords, tokens
- Service account JSON files
- Database connection strings with credentials
- Private keys or certificates

### Use Instead
- Environment variables (via `.env` file, gitignored)
- Your orchestration platform's secret store (Kubernetes Secrets, AWS Secrets Manager, etc.)
- `.env.example` with safe sample values for documentation

### Current Secrets
The following environment variables contain sensitive data:
- `EMAIL_PASSWORD`
- `SENTRY_DSN` (if used)
- `INDEX_DB_URL` (if contains credentials)
- Any API keys in `emailops.llm_runtime` (GCP credentials, etc.)

## Runtime Hardening

### Container Security
- Run as non-root user (UID 10001 "appuser")
- Use minimal base image (`python:3.11-slim`)
- Enable healthchecks for orchestration
- Mount data directories with least privilege

### Network Security
- Restrict egress to necessary endpoints only
- Use TLS/SSL for all external connections
- Implement rate limiting on inbound requests
- Log all authentication attempts

### Code Security
- Input validation on all boundaries
- Path traversal prevention (see `emailops/validators.py`)
- SQL injection prevention (parameterized queries)
- Command injection prevention (avoid shell=True)
- CSV injection prevention (escape formula prefixes)

## Vulnerability Classes to Watch

### Currently Mitigated
✅ Path traversal (validators.py)
✅ Command injection (validators.py)
✅ CSV injection (summarize_email_thread.py)
✅ Control character injection (utils.py)

### Requires Vigilance
⚠️ ReDoS (regex patterns - see ISSUES docs)
⚠️ TOCTOU races (file operations - documented but unfixable at app layer)
⚠️ Memory exhaustion (implement size limits on all external inputs)
⚠️ Rate limiting (implement per-client limits if exposed as service)

## Dependency Security

### Automated Scanning
- Dependabot enabled for weekly updates
- Bandit security scan in CI pipeline
- GitHub Security Advisories monitored

### Manual Review
Review dependencies quarterly for:
- Known CVEs
- Unmaintained packages
- Excessive privileges
- Supply chain risks

## Incident Response

### If Vulnerability Discovered

1. **Assess**: Severity, exploitability, affected versions
2. **Patch**: Develop fix in private branch
3. **Test**: Verify fix doesn't break functionality
4. **Disclose**: Coordinate disclosure timeline
5. **Deploy**: Release patch, notify users
6. **Document**: Post-mortem and lessons learned

### Security Contacts

- Primary: [Maintainer email]
- Secondary: GitHub Security Advisories
- Emergency: [Emergency contact]

## Compliance

### Data Handling
- Email content may contain PII
- Implement data retention policies
- Support data deletion requests
- Log access to sensitive data

### Audit Trail
- All authentication events logged
- Configuration changes logged
- Data access logged (with sanitization)
- Failed operations logged

## Security Checklist for Deployment

- [ ] All secrets in environment variables
- [ ] Container runs as non-root
- [ ] Healthcheck configured
- [ ] Logs don't contain sensitive data
- [ ] Rate limiting implemented
- [ ] Network egress restricted
- [ ] TLS/SSL for external connections
- [ ] Security scan passing (Bandit)
- [ ] Dependencies up to date
- [ ] Backup and recovery tested
- [ ] Incident response plan documented
- [ ] Access controls configured
- [ ] Monitoring and alerting active

## Resources

- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [CWE Top 25](https://cwe.mitre.org/top25/)
- [Python Security Best Practices](https://python.readthedocs.io/en/stable/library/security.html)

---

**Last Updated**: 2025-10-16  
**Next Review**: 2026-01-16
