import sys

try:
    import pgvector.sqlalchemy
except ImportError:
    print("Could not import pgvector.sqlalchemy. Is pgvector installed?")
    sys.exit(1)

print(f"pgvector.sqlalchemy contents: {dir(pgvector.sqlalchemy)}")

# Check for Vector
if hasattr(pgvector.sqlalchemy, "Vector"):
    print("Vector type:", type(pgvector.sqlalchemy.Vector))
else:
    print("Vector type not found.")

# Check for HalfVector
if hasattr(pgvector.sqlalchemy, "HalfVector"):
    from pgvector.sqlalchemy import HalfVector

    print("HalfVector type:", type(HalfVector))
    print("HalfVector doc:", HalfVector.__doc__)
else:
    print("HalfVector not found.")
