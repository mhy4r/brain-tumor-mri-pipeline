# scripts/load_data.py
import pandas as pd
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from database_connection import get_connection

def load_data() -> pd.DataFrame:
    print("[LOAD] Querying database...")
    con = get_connection()

    # Join all tables into one flat DataFrame — same shape your notebook expects
    query = """
    SELECT 
        i.id, 
        i.file_name, 
        i.width, 
        i.height, 
        i.x AS point_x, 
        i.y AS point_y, 
        i.description, 
        t.type_name AS tumor_type, 
        g.grade AS weighting, 
        (t.type_name || ' ' || g.grade) AS class,
        
        -- Aggregates multiple location tags into a single text string
        GROUP_CONCAT(l.location_name, ', ') AS tumor_locations

    FROM images i
    JOIN tumor_types t ON i.type_id = t.id
    JOIN tumor_grades g ON i.grade_id = g.id
    
    -- LEFT JOIN keeps records even if meta["location"] was empty/omitted
    LEFT JOIN image_locations il ON i.id = il.image_id
    LEFT JOIN locations l ON il.location_id = l.id
    
    -- Groups your rows so you get exactly ONE clean sample row per image matrix
    GROUP BY i.id
    """
    
    df = pd.read_sql_query(query, con)
    con.close()

    print(f"[LOAD] Loaded {len(df)} rows.")
    print(f"[LOAD] Columns: {list(df.columns)}")
    print(f"[LOAD] Tumor types  : {sorted(df['tumor_type'].unique())}")
    print(f"[LOAD] Weightings   : {sorted(df['weighting'].unique())}")
    return df

if __name__ == '__main__':
    df = load_data()
    print(df.head())

    # ──────────────────────────────────────────────────────────────────────────
    #  DATA COMPLETENESS AND INTEGRITY TESTS
    # ──────────────────────────────────────────────────────────────────────────
    print(f"\ [TEST] Baseline Row Count Check:")
    print(f"    Expected: 11300 | Found: {len(df)}")
    print("    Match!" if len(df) == 11300 else "    Warning: Count mismatch!")

    print(f"\n [TEST] Null and Empty Value Inspection:")
    for col in df.columns:
        print(f"     - {col.ljust(15)}: {df[col].isnull().sum()} nulls")
        
    print(f"   • Empty string descriptions: {(df['description'].str.strip() == '').sum()}")

    print(f"\n [TEST] Coordinate Integrity Boundaries:")
    out_of_bounds = df[(df['point_x'] > df['width']) | (df['point_y'] > df['height'])].shape[0]
    print(f"   • Out-of-bounds tumor coordinates: {out_of_bounds}")
    print(f"   • X range: [{df['point_x'].min()} to {df['point_x'].max()}] | Y range: [{df['point_y'].min()} to {df['point_y'].max()}]")

    print(f"\n  [TEST] Categorical Value Sane Check:")
    print(f"   • Unique Tumor Types ({len(df['tumor_type'].unique())}): {sorted(df['tumor_type'].unique())}")
    print(f"   • Unique Weightings  ({len(df['weighting'].unique())}): {sorted(df['weighting'].unique())}")
    print(f"   • Unique Class Combinations: {len(df['class'].unique())}")
    print("="*60)
    print("="*60)
    print("="*60)
    print(df.columns.to_list())