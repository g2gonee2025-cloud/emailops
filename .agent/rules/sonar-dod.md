# Definition of Done: SonarQube

You are strictly bound by the SonarQube Quality Gate.

1. **Mandatory Verification**: You MUST verify your work by running: `./scripts/sonar_gate.sh`.
2. **Success Condition**: You MUST NOT mark a task as "Done" or "Completed" unless that script exits with code 0 (Success).
3. **Loop Logic**:
   - If the script fails (Exit code 1):
     - Read the "Conditions causing failure" output.
     - Modify the code to fix the highest priority failure.
     - **Immediately re-run** `./scripts/sonar_gate.sh`.
   - Do NOT ask for user permission to re-run the script. Just do it.
