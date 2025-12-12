import subprocess
from pathlib import Path


def check_command(cmd):
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return f"{cmd}: Return Code {result.returncode}\nStdout: {result.stdout}\nStderr: {result.stderr}\n"
    except Exception as e:
        return f"{cmd}: Exception {e}\n"


output_path = Path("tool_check_output.txt")
with output_path.open("w") as f:
    f.write(check_command("python --version"))
    f.write(check_command("pre-commit --version"))
    f.write(check_command("pytest --version"))
    f.write(check_command("doctl version"))
