---
description: While SonarQube Quality Gate is not OK, fix repo until it is OK
---

**Pre-req:**
- Ensure `SONAR_TOKEN` and `SONAR_HOST_URL` are set (same variables used by SonarSource scan action / SonarScanner).

**Loop (max 10 iterations):**

1. Run: `./scripts/sonar_gate.sh`

2. If it exits 0:
   - **STOP**. Output a summary of what changed and why the gate is now OK.

3. If it exits non-zero:
   - Read the "FAILING_CONDITIONS" and "TOP_ISSUES" output.
   - Fix the minimum necessary code to satisfy the gate:
     - **Bugs/Vulnerabilities**: correct the logic and add/adjust tests.
     - **Coverage-related conditions**: add meaningful tests (do not game coverage).
     - **Code smells**: refactor only what is required; avoid cosmetic churn.
   - Re-run: `./scripts/sonar_gate.sh`
   - Repeat.

**Stop conditions / escalation:**

- If still failing after 10 iterations, **STOP** and output:
  - remaining failing conditions,
  - the specific files/rules most likely blocking,
  - the next 3 concrete code changes to try.

**Notes:**

- The script `sonar_gate.sh` combines scan + quality gate check into one command.
- `sonar_qg.sh` outputs `QUALITY_GATE=OK|ERROR|...`, `FAILING_CONDITIONS:`, and `TOP_ISSUES:`.
- Focus on the highest-severity issues first (BLOCKER > CRITICAL > MAJOR).
