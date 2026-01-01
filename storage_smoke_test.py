import os
import sys
import uuid
from pathlib import Path

# Add backend/src to sys.path
backend_src = Path("/root/workspace/emailops-vertex-ai/backend/src").resolve()
sys.path.append(str(backend_src))

try:
    from cortex.ingestion.s3_source import S3SourceHandler

    print("Testing S3/Spaces connection...")
    with S3SourceHandler() as s3:
        # 1. List
        print("Testing List...")
        folders = list(s3.list_conversation_folders(limit=1))
        print(f"List folders successful. Found {len(folders)} folders.")

        # 2. Upload
        print("Testing Upload...")
        test_content = b"SMOKE TEST CONTENT"
        test_file = Path("storage_smoke_test.txt")
        test_file.write_bytes(test_content)

        test_key = f"smoke_tests/{uuid.uuid4()}.txt"
        s3.upload_file(test_file, test_key)
        print(f"Uploaded test file to {test_key}")

        # 3. Download / Get content
        print("Testing Download/Get...")
        downloaded = s3.get_object_content(test_key)
        if downloaded == test_content:
            print("Download verification: PASSED")
        else:
            print(
                f"Download verification: FAILED (Expected {test_content}, got {downloaded})"
            )
            sys.exit(1)

        # 4. Delete
        print("Testing Delete...")
        s3.delete_object(test_key)
        print("Deleted test file.")

        # Verify deletion
        try:
            s3.get_object_content(test_key)
            print("Deletion verification: FAILED (Object still exists)")
            sys.exit(1)
        except Exception:
            print("Deletion verification: PASSED (Object no longer exists)")

    print("Storage Smoke Test Completed Successfully.")
except Exception as e:
    print(f"Storage Smoke Test FAILED: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)
finally:
    if Path("storage_smoke_test.txt").exists():
        Path("storage_smoke_test.txt").unlink()
