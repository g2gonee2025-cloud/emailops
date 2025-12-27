import json

with open("jules_batch_report.json") as f:
    data = json.load(f)

success = sum(1 for item in data if item.get("status") == "success")
failed = sum(1 for item in data if item.get("status") != "success")

print(f"Total: {len(data)}")
print(f"Success: {success}")
print(f"Failed: {failed}")
