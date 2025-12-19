import sys

# Ensure cortex_cli is importable
sys.path.append("/root/workspace/emailops-vertex-ai/cli/src")

try:
    from cortex_cli import tui

    print("Successfully imported tui")
except ImportError as e:
    print(f"Failed to import tui: {e}")
    sys.exit(1)
except Exception as e:
    print(f"Error during import: {e}")
    sys.exit(1)
