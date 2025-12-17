import json
import sys
from pathlib import Path

root = Path("temp_s3_validation")
manifests = list(root.rglob("manifest.json"))

if not manifests:
    print("No manifests found")
    sys.exit(1)

target = manifests[0]
print(f"Inspecting: {target}")

try:
    data = json.loads(target.read_text())
    print("Keys found:", list(data.keys()))

    if "messages" in data:
        print(f"Messages count: {len(data['messages'])}")
        print("First message sample:", json.dumps(data["messages"][0], indent=2))
    else:
        print("WARNING: 'messages' key NOT FOUND in this manifest.")

except Exception as e:
    print(f"Error reading manifest: {e}")
