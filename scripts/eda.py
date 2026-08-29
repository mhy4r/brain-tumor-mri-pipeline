# scripts/eda.py
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from database_connection import get_connection

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# ── 1. Load ──────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("BRAIN TUMOR MRI — EXPLORATORY DATA ANALYSIS")
print("="*60)

con = get_connection()

df = pd.read_sql_query("""
    SELECT
        i.id,
        i.file_name,
        i.width,
        i.height,
        i.x          AS point_x,
        i.y          AS point_y,
        i.description,
        t.type_name  AS tumor_type,
        g.grade      AS weighting,
        (t.type_name || ' ' || g.grade) AS class,
        GROUP_CONCAT(l.location_name, ', ') AS tumor_locations
    FROM images i
    JOIN tumor_types  t ON i.type_id  = t.id
    JOIN tumor_grades g ON i.grade_id = g.id
    LEFT JOIN image_locations il ON i.id = il.image_id
    LEFT JOIN locations l ON il.location_id = l.id
    GROUP BY i.id
""", con)

# Load locations separately
df_loc = pd.read_sql_query("""
    SELECT il.image_id, l.location_name
    FROM image_locations il
    JOIN locations l ON il.location_id = l.id
""", con)
con.close()

print(f"\n[EDA] Total images loaded : {len(df)}")
print(f"[EDA] Total columns       : {len(df.columns)}")
print(f"[EDA] Columns             : {list(df.columns)}")

# ── 2. General structure ─────────────────────────────────────────────────────
print("\n" + "-"*60)
print("GENERAL STRUCTURE")
print("-"*60)
print(df.dtypes.to_string())
print(f"\nFirst 3 rows:")
print(df.head(3).to_string())

# ── 3. Missing values ─────────────────────────────────────────────────────────
print("\n" + "-"*60)
print("MISSING VALUES")
print("-"*60)
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_report = pd.DataFrame({'missing_count': missing, 'missing_%': missing_pct})
missing_report = missing_report[missing_report['missing_count'] > 0]
if len(missing_report):
    print(missing_report.to_string())
else:
    print("No missing values found.")

empty_desc = (df['description'].str.strip() == '').sum()
print(f"Empty description strings : {empty_desc} ({empty_desc/len(df)*100:.1f}%)")

# ── 4. Images per tumor type ──────────────────────────────────────────────────
print("\n" + "-"*60)
print("IMAGES PER TUMOR TYPE")
print("-"*60)
tumor_counts = df['tumor_type'].value_counts()
for tumor, count in tumor_counts.items():
    pct = count / len(df) * 100
    bar = "█" * int(pct / 2)
    print(f"  {tumor:<22} {count:>5}  ({pct:5.1f}%)  {bar}")

# ── 5. Images per MRI weighting ───────────────────────────────────────────────
print("\n" + "-"*60)
print("IMAGES PER MRI WEIGHTING")
print("-"*60)
weight_counts = df['weighting'].value_counts()
for w, count in weight_counts.items():
    pct = count / len(df) * 100
    bar = "█" * int(pct / 2)
    print(f"  {w:<10} {count:>5}  ({pct:5.1f}%)  {bar}")

# ── 6. Images per full class (tumor * weighting) ─────────────────────────────
print("\n" + "-"*60)
print("IMAGES PER FULL CLASS (TUMOR * WEIGHTING)")
print("-"*60)
class_counts = df['class'].value_counts()
for cls, count in class_counts.items():
    print(f"  {cls:<30} {count:>5}")

# ── 7. Pivot table: tumor * weighting ────────────────────────────────────────
print("\n" + "-"*60)
print("PIVOT: TUMOR TYPE * WEIGHTING (image counts)")
print("-"*60)
pivot = df.pivot_table(
    index='tumor_type', columns='weighting',
    values='id', aggfunc='count', fill_value=0
)
print(pivot.to_string())

# ── 8. Patient count ──────────────────────────────────────────────────────────
# A "patient" = unique case extracted from filename
# Strip weighting prefix and slice number to get case name
import re

def extract_case(filename):
    name = re.sub(r'^T1C\+\s*[-–]\s*|^T1\s*[-–]\s*|^T2\s*[-–]\s*', '', filename)
    name = re.sub(r'\s*\d+(_aug\d+)?\.jpg$', '', name, flags=re.IGNORECASE).strip()
    return name

df['case_name'] = df['file_name'].apply(
    lambda p: extract_case(os.path.basename(p.replace('\\', '/')))
)

print("\n" + "-"*60)
print("PATIENT (UNIQUE CASE) COUNTS")
print("-"*60)
total_patients = df.groupby(['tumor_type', 'case_name']).ngroups
print(f"\nTotal unique patients across all classes: {total_patients}")

print("\nPatients per tumor type:")
patients_per_type = (
    df.groupby('tumor_type')['case_name']
    .nunique()
    .sort_values(ascending=False)
)
for tumor, n_patients in patients_per_type.items():
    n_images = tumor_counts[tumor]
    avg_imgs = n_images / n_patients
    print(f"  {tumor:<22} {n_patients:>3} patients  |  "
          f"{n_images:>5} images  |  {avg_imgs:>5.1f} avg images/patient")

# ── 9. Lesion point statistics ────────────────────────────────────────────────
print("\n" + "-"*60)
print("LESION POINT STATISTICS")
print("-"*60)
normal_mask = df['tumor_type'] == 'Normal'
print(f"\nNormal scans (point = image center): {normal_mask.sum()}")
print(f"Tumor scans  (point = lesion center): {(~normal_mask).sum()}")
print(f"\npoint_x stats:")
print(df['point_x'].describe().round(2).to_string())
print(f"\npoint_y stats:")
print(df['point_y'].describe().round(2).to_string())

oob = df[(df['point_x'] > df['width']) | (df['point_y'] > df['height'])]
print(f"\nOut-of-bounds lesion points: {len(oob)}")

df['lesion_offset'] = np.sqrt(
    (df['point_x'] - df['width']  / 2) ** 2 +
    (df['point_y'] - df['height'] / 2) ** 2
)
print(f"\nLesion offset from center (pixels):")
print(df.groupby('tumor_type')['lesion_offset'].mean().round(1)
      .sort_values(ascending=False).to_string())

# ── 10. Augmented vs original ────────────────────────────────────────────────
print("\n" + "-"*60)
print("AUGMENTED VS ORIGINAL IMAGES")
print("-"*60)
is_aug = df['file_name'].str.contains('_aug', case=False, na=False)
print(f"  Original images  : {(~is_aug).sum()}")
print(f"  Augmented images : {is_aug.sum()}")
if is_aug.sum() > 0:
    aug_ratio = is_aug.sum() / (~is_aug).sum()
    print(f"  Augmentation ratio: {aug_ratio:.1f}x")

# ── 11. Location distribution ────────────────────────────────────────────────
print("\n" + "-"*60)
print("LESION LOCATION DISTRIBUTION")
print("-"*60)
if len(df_loc):
    loc_counts = df_loc['location_name'].value_counts()
    print(f"Total location tags: {len(df_loc)}")
    print(f"Unique locations   : {df_loc['location_name'].nunique()}")
    print("\nTop 20 locations:")
    print(loc_counts.head(20).to_string())
else:
    print("No location data found.")

print("\n[EDA] Tumor Type Distribution by Top 10 Locations:")
# Explode the comma-separated locations to count individual pairs accurately
df_exploded = df.assign(location=df['tumor_locations'].fillna('Unspecified').str.split(', ')).explode('location')
top_10_locs = df_exploded['location'].value_counts().head(10).index

loc_tumor_pivot = pd.crosstab(df_exploded['location'].values, df_exploded['tumor_type'].values)
loc_tumor_pivot.index.name = 'location'
loc_tumor_pivot.columns.name = 'tumor_type'
    
print(loc_tumor_pivot.loc[top_10_locs].to_string())

# ── 12. Image dimension check ────────────────────────────────────────────────
print("\n" + "-"*60)
print("IMAGE DIMENSIONS")
print("-"*60)
dim_counts = df.groupby(['width', 'height']).size().reset_index(name='count')
print(dim_counts.to_string(index=False))

# ── 13. Class imbalance ───────────────────────────────────────────────────────
print("\n" + "-"*60)
print("CLASS IMBALANCE SUMMARY")
print("-"*60)
print(f"  Most common tumor  : {tumor_counts.idxmax()} ({tumor_counts.max()} images)")
print(f"  Least common tumor : {tumor_counts.idxmin()} ({tumor_counts.min()} images)")
print(f"  Imbalance ratio    : {tumor_counts.max() / tumor_counts.min():.2f}x")

# ── 14. Plots ────────────────────────────────────────────────────────────────
print("\n" + "-"*60)
print("SAVING PLOTS")
print("-"*60)
os.makedirs('eda_output', exist_ok=True)

# Plot 1: images per tumor type
fig, ax = plt.subplots(figsize=(12, 5))
tumor_counts.plot(kind='bar', ax=ax, color='steelblue', edgecolor='black')
ax.set_title('Images per Tumor Type')
ax.set_xlabel('Tumor Type')
ax.set_ylabel('Count')
ax.tick_params(axis='x', rotation=45)
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() + p.get_width()/2, p.get_height()),
                ha='center', va='bottom', fontsize=8)
plt.tight_layout()
plt.savefig('eda_output/01_images_per_tumor.png', dpi=150)
plt.close()
print("  Saved: eda_output/01_images_per_tumor.png")

# Plot 2: images per weighting
fig, ax = plt.subplots(figsize=(6, 4))
weight_counts.plot(kind='bar', ax=ax, color='coral', edgecolor='black')
ax.set_title('Images per MRI Weighting')
ax.set_xlabel('Weighting')
ax.set_ylabel('Count')
ax.tick_params(axis='x', rotation=0)
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() + p.get_width()/2, p.get_height()),
                ha='center', va='bottom', fontsize=9)
plt.tight_layout()
plt.savefig('eda_output/02_images_per_weighting.png', dpi=150)
plt.close()
print("  Saved: eda_output/02_images_per_weighting.png")

# Plot 3: heatmap tumor * weighting
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(pivot, annot=True, fmt='d', cmap='Blues', ax=ax)
ax.set_title('Image Count: Tumor Type * MRI Weighting')
plt.tight_layout()
plt.savefig('eda_output/03_pivot_heatmap.png', dpi=150)
plt.close()
print("  Saved: eda_output/03_pivot_heatmap.png")

# Plot 4: patients per tumor type
fig, ax = plt.subplots(figsize=(12, 5))
patients_per_type.plot(kind='bar', ax=ax, color='mediumseagreen', edgecolor='black')
ax.set_title('Unique Patients per Tumor Type')
ax.set_xlabel('Tumor Type')
ax.set_ylabel('Number of Patients')
ax.tick_params(axis='x', rotation=45)
for p in ax.patches:
    ax.annotate(str(int(p.get_height())),
                (p.get_x() + p.get_width()/2, p.get_height()),
                ha='center', va='bottom', fontsize=8)
plt.tight_layout()
plt.savefig('eda_output/04_patients_per_tumor.png', dpi=150)
plt.close()
print("  Saved: eda_output/04_patients_per_tumor.png")

# Plot 5: lesion point scatter
fig, ax = plt.subplots(figsize=(6, 6))
ax.hist2d(df['point_x'], df['point_y'], bins=40, cmap='hot')
ax.set_title('Lesion Point Distribution (all classes)')
ax.set_xlabel('X'); ax.set_ylabel('Y')
ax.invert_yaxis()
plt.tight_layout()
plt.savefig('eda_output/05_lesion_point_heatmap.png', dpi=150)
plt.close()
print("  Saved: eda_output/05_lesion_point_heatmap.png")

# Plot 6: lesion offset per tumor type
fig, ax = plt.subplots(figsize=(12, 5))
df[df['tumor_type'] != 'Normal'].boxplot(
    column='lesion_offset', by='tumor_type', ax=ax
)
ax.set_title('Lesion Offset from Image Center by Tumor Type')
ax.set_xlabel('Tumor Type')
ax.set_ylabel('Offset (pixels)')
plt.suptitle('')
ax.tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.savefig('eda_output/06_lesion_offset_boxplot.png', dpi=150)
plt.close()
print("  Saved: eda_output/06_lesion_offset_boxplot.png")

# Plot 7: Stacked Bar Chart of Tumor Types over Top 10 Locations
if 'df_exploded' in locals() and len(df_exploded):
    fig, ax = plt.subplots(figsize=(14, 7))
    loc_tumor_pivot.loc[top_10_locs].plot(kind='bar', stacked=True, ax=ax, cmap='tab10', edgecolor='black')
    ax.set_title('Tumor Type Distribution across Top 10 Locations')
    ax.set_xlabel('Anatomical Location')
    ax.set_ylabel('Image Count')
    ax.tick_params(axis='x', rotation=45)
    plt.legend(title='Tumor Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('eda_output/07_tumor_distribution_per_location.png', dpi=150)
    plt.close()
    print("  Saved: eda_output/07_tumor_distribution_per_location.png")

print("\n" + "="*60)
print("EDA COMPLETE")
print("="*60)