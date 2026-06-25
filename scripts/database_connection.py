# scripts/database_connection.py
import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'database','brain_tumor_db.sqlite')

def get_connection():
    con = sqlite3.connect(DB_PATH)
    print(f"[DB_CONN] Connected to: {os.path.abspath(DB_PATH)}")
    return con

if __name__ == '__main__':
    con = get_connection()
    cur = con.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [r[0] for r in cur.fetchall()]
    print(f"[DB_CONN] Tables found: {tables}")
    con.close()