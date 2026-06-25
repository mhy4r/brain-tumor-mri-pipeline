# pipeline.py
import subprocess, sys, os

# Make sure DB exists before running pipeline
DB_PATH = 'brain_tumor_db.sqlite'
if not os.path.exists(DB_PATH):
    print(f"[PIPELINE] DB not found at {DB_PATH}")
    print("[PIPELINE] Run: python scripts/build_database.py  first")
    sys.exit(1)

steps = [
    ("Load Data",           ["python", "scripts/load_data.py"]),
    ("Preprocess",          ["python", "scripts/preprocess.py"]),
    ("Feature Engineering", ["python", "scripts/feature_engineering.py"]),
]

print("=" * 50)
print("BRAIN TUMOR PIPELINE — START")
print("=" * 50)

for name, cmd in steps:
    print(f"\n>>> Step: {name}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"[PIPELINE] FAILED at: {name}")
        sys.exit(result.returncode)
    print(f"[PIPELINE] OK: {name}")

print("\n" + "=" * 50)
print("PIPELINE COMPLETE")
print("=" * 50)