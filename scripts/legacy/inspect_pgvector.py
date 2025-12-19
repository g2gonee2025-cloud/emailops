import pgvector.sqlalchemy

print(dir(pgvector.sqlalchemy))
print("Vector type:", type(pgvector.sqlalchemy.Vector))
try:
    print("HalfVector type:", type(pgvector.sqlalchemy.HalfVector))
except Exception as e:
    print(f"HalfVector not found: {e}")

from pgvector.sqlalchemy import HalfVector  # noqa: E402

print("HalfVector doc:", HalfVector.__doc__)
