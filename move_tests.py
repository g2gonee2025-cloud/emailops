import shutil
from pathlib import Path

try:
    tests_path = Path("tests")
    backend_tests_path = Path("backend/tests")

    if tests_path.exists() and not backend_tests_path.exists():
        shutil.move(str(tests_path), str(backend_tests_path))
        print("Moved tests to backend/tests")
    elif tests_path.exists() and backend_tests_path.exists():
        print("Both exist, merging not implemented safely")
    else:
        print("Source tests not found or something else")
except Exception as e:
    print(f"Error: {e}")
