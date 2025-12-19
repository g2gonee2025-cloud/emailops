import sys

# Mock paths
sys.path.append("backend/src")

try:
    print("Importing cortex.db.models...")
    from cortex.db.models import Chunk

    print("Chunk model imported.")
    print(f"Embedding column type: {Chunk.embedding.type}")
except Exception as e:
    print(f"Error loading models: {e}")
    import traceback

    traceback.print_exc()
