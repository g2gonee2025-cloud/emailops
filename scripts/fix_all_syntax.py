import os
import re

# Target the site-packages directory
ROOT = "/root/workspace/emailops-vertex-ai/.venv/lib/python3.13/site-packages"

# Regex to match __all__ = ( ... ]
# Captures group 1: __all__ = ( ...
# Matches closing ]
# flags=re.DOTALL to match newlines
PATTERN = re.compile(r"(__all__\s*=\s*\((?:[^\]]|\n)*?)\]", re.DOTALL)

count = 0
for root, _, files in os.walk(ROOT):
    for file in files:
        if file.endswith(".py"):
            path = os.path.join(root, file)
            try:
                with open(path, encoding="utf-8", errors="ignore") as f:
                    content = f.read()

                if "__all__" not in content:
                    continue

                new_content = PATTERN.sub(r"\1)", content)

                if new_content != content:
                    print(f"Fixing {path}")
                    with open(path, "w", encoding="utf-8") as f:
                        f.write(new_content)
                    count += 1
            except Exception as e:
                print(f"Error processing {path}: {e}")

print(f"Total fixed files: {count}")
