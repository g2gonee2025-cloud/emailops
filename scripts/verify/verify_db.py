import os

import psycopg2
from dotenv import load_dotenv

load_dotenv()

db_url = os.getenv("DB_URL")
print(f"Connecting to: {db_url.split('@')[1]}")  # Mask auth

try:
    conn = psycopg2.connect(db_url)
    print("✅ Connected successfully")
    cur = conn.cursor()
    cur.execute("SELECT 1")
    print(f"✅ Query result: {cur.fetchone()}")
    conn.close()
except Exception as e:
    print(f"❌ Connection failed: {e}")
