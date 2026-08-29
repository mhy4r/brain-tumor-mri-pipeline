# %%writefile scripts/verify_db.py
import sqlite3
import os
import pandas as pd

# Use your existing connection logic
DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'brain_tumor_db.sqlite')

def verify_database():
    if not os.path.exists(DB_PATH):
        print(f"Error: Database file not found at {os.path.abspath(DB_PATH)}")
        return

    conn = sqlite3.connect(DB_PATH)
    
    print("=" * 60)
    print("DATABASE AUDIT & INTEGRITY CHECK")
    print("=" * 60)
    
    # 1. Check Row Counts across all tables
    print("\ 1. Row Counts per Table:")
    tables = ['tumor_types', 'tumor_grades', 'images', 'locations', 'image_locations']
    for table in tables:
        count = pd.read_sql_query(f"SELECT COUNT(*) as cnt FROM {table}", conn)['cnt'].iloc[0]
        print(f"   • {table.ljust(18)} : {count} records")

    # 2. Inspect Unique Classes Extracted
    print("\n 2. Unique Tumor Types Found:")
    types_df = pd.read_sql_query("SELECT * FROM tumor_types", conn)
    print(types_df.to_string(index=False))

    print("\n 3. Unique Tumor Grades Found:")
    grades_df = pd.read_sql_query("SELECT * FROM tumor_grades", conn)
    print(grades_df.to_string(index=False))

    # 4. Spot Check a Joined Record (Testing your Foreign Keys)
    print("\n 4. Relational Join Spot Check (Top 3 Images with Labels):")
    query = """
        SELECT 
            i.id, 
            i.file_name, 
            i.width, 
            i.height, 
            i.x, 
            i.y,
            t.type_name, 
            g.grade
        FROM images i
        JOIN tumor_types t ON i.type_id = t.id
        JOIN tumor_grades g ON i.grade_id = g.id
        LIMIT 3;
    """
    sample_df = pd.read_sql_query(query, conn)
    print(sample_df.to_string(index=False))

    # 5. Check for Outliers or Null Values
    print("\n 5. Data Integrity Checks:")
    null_images = pd.read_sql_query("SELECT COUNT(*) as cnt FROM images WHERE type_id IS NULL OR grade_id IS NULL", conn)['cnt'].iloc[0]
    print(f"   • Images missing a valid foreign key mapping : {null_images}")
    
    invalid_coords = pd.read_sql_query("SELECT COUNT(*) as cnt FROM images WHERE x > width OR y > height", conn)['cnt'].iloc[0]
    print(f"   • Images with out-of-bounds coordinates     : {invalid_coords}")

    print("=" * 60)
    conn.close()

if __name__ == '__main__':
    verify_database()