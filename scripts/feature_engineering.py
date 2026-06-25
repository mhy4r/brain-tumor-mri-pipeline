# scripts/feature_engineering.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from load_data import load_data
from preprocess import preprocess

TUMOR2IDX = {
    'Normal':0, 'Astrocytoma':1, 'Ependymoma':2, 'Glioma':3,
    'Hemangiopericytoma':4, 'Meningioma':5, 'Neurocytoma':6,
    'Oligodendroglioma':7, 'Schwannoma':8, 'Other':9
}
WEIGHT2IDX = {'T1': 0, 'T2': 1, 'T1C+': 2}

def run_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    print("\n[FE] Starting feature engineering...")

    # 1. Ordinal encoding
    df['tumor_idx']  = df['tumor_type'].map(TUMOR2IDX)
    df['weight_idx'] = df['weighting'].map(WEIGHT2IDX)
    print(f"[FE] Ordinal encoded: tumor_idx, weight_idx")

    # 2. One-hot encoding for tumor type and weighting
    tumor_dummies  = pd.get_dummies(df['tumor_type'], prefix='tumor')
    weight_dummies = pd.get_dummies(df['weighting'],  prefix='weight')
    df = pd.concat([df, tumor_dummies, weight_dummies], axis=1)
    print(f"[FE] One-hot encoded tumor_type -> {list(tumor_dummies.columns)}")
    print(f"[FE] One-hot encoded weighting  -> {list(weight_dummies.columns)}")

    # 3. Spatial features
    df['lesion_offset'] = np.sqrt(
        (df['point_x'] - df['width']  / 2) ** 2 +
        (df['point_y'] - df['height'] / 2) ** 2
    )
    df['lesion_offset_norm'] = df['lesion_offset'] / np.sqrt(
        (df['width']  / 2) ** 2 +
        (df['height'] / 2) ** 2
    )
    df['is_central'] = (df['lesion_offset_norm'] < 0.2).astype(int)
    print("[FE] Spatial features: lesion_offset, lesion_offset_norm, is_central")

    # 4. Boolean flags
    df['is_normal']    = (df['tumor_type'] == 'Normal').astype(int)
    print(f"[FE] Flags: is_normal ({df['is_normal'].sum()} rows)")

    # 5. Text feature: description length + TF-IDF top 20 terms
    df['desc_length'] = df['description'].str.len().fillna(0).astype(int)
    print(f"[FE] Text feature: desc_length (mean={df['desc_length'].mean():.1f} chars)")

    non_empty = df['description'].fillna('').str.strip()
    if non_empty.str.len().sum() > 0:
        tfidf = TfidfVectorizer(max_features=20, stop_words='english')
        tfidf_matrix = tfidf.fit_transform(non_empty)
        tfidf_df = pd.DataFrame(
            tfidf_matrix.toarray(),
            columns=[f'tfidf_{t}' for t in tfidf.get_feature_names_out()],
            index=df.index
        )
        df = pd.concat([df, tfidf_df], axis=1)
        print(f"[FE] TF-IDF features added: {list(tfidf_df.columns)}")
    else:
        print("[FE] Descriptions empty — TF-IDF skipped.")

    # 6. Null check after engineering
    key_cols = ['tumor_idx', 'weight_idx', 'lesion_offset']
    
    loc_cols = [c for c in df.columns if c.startswith('loc_')]
    check_cols = key_cols + loc_cols

    null_counts = df[check_cols].isnull().sum()
    null_counts_filtered = null_counts[null_counts > 0]
    
    print("\n[FE] Checking location features...")
    print(f"[FE] Total location feature metrics tracked: {len(loc_cols)}")
    if len(loc_cols) > 0:
        # Log top active location stats to verify distribution
        active_counts = df[loc_cols].sum().sort_values(ascending=False)
        print(f"[FE] Top 3 most frequent tumor locations:\n{active_counts.head(3).to_string()}")
    
    print(f"[FE] Null check on key features and location flags:\n{null_counts_filtered.to_string() if len(null_counts_filtered) else '  All clear! No nulls found.'}")

    out = 'processed_features.csv'
    df.to_csv(out, index=False)
    print(f"[FE] Saved enriched DataFrame to {out} — {len(df)} rows, {len(df.columns)} columns.")
    return df

if __name__ == '__main__':
    df = load_data()
    df = preprocess(df)
    df = run_feature_engineering(df)
    print(df[['tumor_type','tumor_idx','weighting','weight_idx',
              'lesion_offset_norm','is_central','is_normal']].head())
    print(df.columns.to_list())
    print("="*60)

    loc_preview = [c for c in df.columns if c.startswith('loc_')][:4]
    print("\n=== PIPELINE VERIFICATION PREVIEW ===")
    preview_cols = ['tumor_type', 'tumor_idx', 'weighting', 'weight_idx', 'lesion_offset_norm', 'is_central'] + loc_preview
    print(df[preview_cols].head())
    print(f"\nTotal structural columns in final tensor: {len(df.columns)}")