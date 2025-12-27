import sys
from pathlib import Path

# Mock paths
backend_src = Path(__file__).resolve().parents[2] / "backend" / "src"
if backend_src.is_dir():
    sys.path.insert(0, str(backend_src))

try:
    print("Importing cortex.db.models...")
    from cortex.db.models import Chunk

    print("Chunk model imported.")
    print(f"Embedding column type: {Chunk.embedding.type}")
except (ImportError, AttributeError) as e:
    print(f"Error loading models: {e}")
    import traceback

    traceback.print_exc()
