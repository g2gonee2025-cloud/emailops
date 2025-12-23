---
description: Fix code until SonarQube Quality Gate passes
---

**Pre-req:**
- Ensure `SONAR_TOKEN` and `SONAR_HOST_URL` are set.
- Ensure `scripts/sonar_gate.sh` exists and is executable.

**Loop (Maximum 15 iterations):**

1. // turbo
   Run: `./scripts/sonar_gate.sh`

2. If it exits 0:
   - **STOP**. Output a summary of what changed and why the gate is now OK.

3. If it exits non-zero:
   - Read the "FAILING_CONDITIONS", "TOP_ISSUES", and output logs.
   - Fix the minimum necessary code to satisfy the gate:
     - **Bugs/Vulnerabilities**: correct the logic and add/adjust tests (highest severity first).
     - **Coverage-related conditions**: add meaningful tests (do not game coverage).
     - **Code smells**: refactor only what is required; avoid cosmetic churn.
   - Re-run the loop from Step 1.

**Stop conditions / escalation:**
- If still failing after 15 iterations, **STOP** and output:
  - remaining failing conditions.
  - the specific files/rules most likely blocking.
  - the next 3 concrete code changes to try.

**Notes:**
- `sonar_gate.sh` combines scan + quality gate check into one command.
- Focus on the highest-severity issues first (BLOCKER > CRITICAL > MAJOR).
