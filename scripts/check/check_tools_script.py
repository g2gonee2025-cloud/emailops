import subprocess
import sys
from pathlib import Path


def check_command(cmd_list):
    """Runs a command and captures its output."""
    cmd_str = " ".join(cmd_list)
    try:
        # Using check=False to prevent CalledProcessError on non-zero exit codes,
        # as we are capturing the return code manually.
        result = subprocess.run(
            cmd_list, capture_output=True, text=True, check=False, timeout=60
        )
        return f"{cmd_str}: Return Code {result.returncode}\nStdout: {result.stdout}\nStderr: {result.stderr}\n"
    except FileNotFoundError:
        return f"{cmd_str}: ERROR: Command not found.\n"
    except subprocess.TimeoutExpired:
        return f"{cmd_str}: ERROR: Command timed out after 60 seconds.\n"
    except PermissionError:
        return f"{cmd_str}: ERROR: Permission denied to execute.\n"
    except Exception as e:
        return f"{cmd_str}: ERROR: An unexpected exception occurred: {e}\n"


def main():
    """Checks for the presence and version of essential command-line tools."""
    commands_to_check = [
        ["python", "--version"],
        ["pre-commit", "--version"],
        ["pytest", "--version"],
        ["doctl", "version"],
    ]

    output_path = Path("tool_check_output.txt")

    try:
        with output_path.open("w") as f:
            for cmd in commands_to_check:
                f.write(check_command(cmd))
    except OSError as e:
        print(f"Failed to write to {output_path}: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Tool check complete. Output written to {output_path}")


if __name__ == "__main__":
    main()
