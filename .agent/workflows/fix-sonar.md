---
description: Fix code until SonarQube Quality Gate passes
---
Pre-req: Ensure SONAR_TOKEN and SONAR_HOST_URL are set.

Loop (Maximum 15 iterations):
1) // turbo Run `./scripts/sonar_gate.sh`
2) If exit code is 0:
   - STOP. Summarize the fixes applied.
3) If exit code is 1:
   - Analyze the "FAILING CONDITIONS" from the output.
   - Perform the minimal code fix required to satisfy the metric (e.g., add test coverage, fix code smell, fix bug).
   - Go back to Step 1.
