# scripts/preprocess.py
import pandas as pd
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from load_data import load_data

VALID_TUMORS = {
    'Normal', 'Astrocytoma', 'Ependymoma', 'Glioma',
    'Hemangiopericytoma', 'Meningioma', 'Neurocytoma',
    'Oligodendroglioma', 'Schwannoma', 'Other'
}
VALID_WEIGHTS = {'T1', 'T1C+', 'T2'}

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    print("\n[PREP] Starting preprocessing...")
    print(f"[PREP] Input rows: {len(df)}")

    # 1. Missing values
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    print(f"[PREP] Missing values:\n{missing.to_string() if len(missing) else '  none'}")
    df['description'] = df['description'].fillna('')
    print("[PREP] Filled missing descriptions with empty string.")

    # 2. Invalid labels
    bad_tumor  = ~df['tumor_type'].isin(VALID_TUMORS)
    bad_weight = ~df['weighting'].isin(VALID_WEIGHTS)
    print(f"[PREP] Rows with unknown tumor type  : {bad_tumor.sum()}")
    print(f"[PREP] Rows with unknown weighting   : {bad_weight.sum()}")
    df = df[~bad_tumor & ~bad_weight].copy()
    print(f"[PREP] Rows after dropping invalid labels: {len(df)}")

    # 3. Out-of-bounds lesion points — log but keep (minor spatial outliers)
    oob = df[(df['point_x'] > df['width']) | (df['point_y'] > df['height'])]
    print(f"[PREP] Out-of-bounds lesion points: {len(oob)} (kept — minor outliers)")

    # 4. Duplicate file names
    dupes = df['file_name'].duplicated().sum()
    print(f"[PREP] Duplicate file_name entries: {dupes}")

    # 5. Normalize point coordinates to [0, 1]
    df['point_x_norm'] = (df['point_x'] / df['width']).clip(0, 1)
    df['point_y_norm'] = (df['point_y'] / df['height']).clip(0, 1)
    print("[PREP] Normalized point_x/y to [0,1] -> point_x_norm, point_y_norm")

    out = 'preprocessed_data.csv'
    df.to_csv(out, index=False)
    print(f"[PREP] Saved to {out} — {len(df)} rows.")
    return df

if __name__ == '__main__':
    df = load_data()
    df = preprocess(df)
    print(df[['tumor_type', 'weighting', 'point_x_norm', 'point_y_norm']].head())