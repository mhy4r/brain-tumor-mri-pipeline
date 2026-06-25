# Brain Tumor MRI Classification Pipeline

A multi-task deep learning pipeline for classifying brain tumor types and MRI weightings
from MRI scans. Built with PyTorch, SQLite, and GitHub Actions CI/CD.

---

## Dataset

This project uses the [Brain Tumor MRI Images — 30 Classes](https://www.kaggle.com/datasets/fernando2rad/brain-tumor-mri-images-30-classes) dataset from Kaggle.

- 11,300 MRI images at 512×512 pixels
- 10 tumor types: Astrocytoma, Ependymoma, Glioma, Hemangiopericytoma, Meningioma, Neurocytoma, Normal, Oligodendroglioma, Other, Schwannoma
- 3 MRI weightings: T1, T1C+, T2
- Each image includes lesion center coordinates (x, y) in metadata

The dataset is **not included** in this repository due to size and licensing.

**To download via Kaggle API:**
```bash
pip install kaggle
# Place your kaggle.json in ~/.kaggle/
# Get it from: https://www.kaggle.com/settings → API → Create New Token
kaggle datasets download -d fernando2rad/brain-tumor-mri-images-30-classes --unzip
```

**Or download manually** from the Kaggle page and extract so the structure looks like:
project/

├── dataset/

│   ├── Astrocytoma T1/

│   ├── Astrocytoma T1C+/

│   ├── Glioma T1/

│   └── ...

└── DATA.json

---

## Project Structure
project/

├── .github/

│   └── workflows/

│       └── ci.yml                  # GitHub Actions CI/CD pipeline

├── database/

│   └── brain_tumor_db.sqlite       # SQLite database (generated locally)

├── scripts/

│   ├── database_connection.py      # DB connection helper used by all scripts

│   ├── import_to_database.py       # One-time ETL: DATA.json → SQLite

│   ├── load_data.py                # Queries DB into a Pandas DataFrame

│   ├── preprocess.py               # Cleans and normalizes data

│   └── feature_engineering.py     # Creates new features and saves CSV

├── pipeline.py                     # Main runner: executes all scripts in order

├── p2-work.ipynb                   # Model training and evaluation notebook

├── preprocessed_data.csv           # Output of preprocess.py

├── processed_features.csv          # Output of feature_engineering.py

├── requirements.txt                # Python dependencies

└── README.md

---

## Setup

**1. Clone the repository**
```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Download the dataset** (see Dataset section above)

**4. Build the database** (run once)
```bash
python scripts/import_to_database.py
```

This reads `DATA.json` and populates `database/brain_tumor_db.sqlite` with a normalized
schema across 5 tables: `images`, `tumor_types`, `tumor_grades`, `locations`, `image_locations`.

---

## Running the Pipeline

```bash
python pipeline.py
```

This sequentially runs:

| Step | Script | Output |
|------|--------|--------|
| 1 | `scripts/load_data.py` | Loads all records from DB into DataFrame |
| 2 | `scripts/preprocess.py` | Cleans labels, handles missing values, normalizes coordinates → `preprocessed_data.csv` |
| 3 | `scripts/feature_engineering.py` | Encodes labels, computes spatial features, TF-IDF on descriptions → `processed_features.csv` |

---

## Database Schema
tumor_types          tumor_grades

──────────           ────────────

id (PK)              id (PK)

type_name            grade
images

──────────────────────────────

id (PK)

file_name

width, height

x, y               ← lesion center coordinates

description

type_id (FK → tumor_types)

grade_id (FK → tumor_grades)
locations            image_locations

─────────            ───────────────

id (PK)              image_id (FK → images)

location_name        location_id (FK → locations)

---

## Feature Engineering

The following features are created in `scripts/feature_engineering.py`:

| Feature | Description |
|---------|-------------|
| `tumor_idx` | Ordinal encoding of tumor type (0–9) |
| `weight_idx` | Ordinal encoding of MRI weighting (0–2) |
| `tumor_*` | One-hot encoded tumor type columns |
| `weight_*` | One-hot encoded weighting columns |
| `point_x_norm` | Lesion X coordinate normalized to [0, 1] |
| `point_y_norm` | Lesion Y coordinate normalized to [0, 1] |
| `lesion_offset` | Euclidean distance of lesion from image center |
| `lesion_offset_norm` | Lesion offset normalized by image diagonal |
| `is_central` | 1 if lesion is within 20% of image center |
| `is_normal` | 1 if scan has no tumor (Normal class) |
| `is_augmented` | 1 if image is an augmented copy |
| `desc_length` | Character length of description text |
| `tfidf_*` | Top 20 TF-IDF terms from description text |

<!-- ---

## Model

The model (`p2-work.ipynb`) is a multi-task CNN (`CustomBrainNet`) with three output heads:

- **Tumor head** — 10-class classification of tumor type
- **Weighting head** — 3-class classification of MRI weighting (T1 / T1C+ / T2)
- **Heatmap head** — Gaussian heatmap regression to localize the lesion

Training uses lesion center coordinates from the metadata as spatial supervision signal. -->

---

## CI/CD

GitHub Actions runs automatically on every push or pull request to `main`.

The pipeline:
1. Sets up Python 3.12
2. Installs all dependencies from `requirements.txt`
3. Creates a minimal synthetic SQLite database (no dataset download needed in CI)
4. Runs `pipeline.py` to verify all scripts execute without errors

See `.github/workflows/ci.yml` for the full configuration.

---

## Requirements

See `requirements.txt`. Key dependencies:

- `torch` — model training
<!-- - `torchvision` — EfficientNet backbone -->
<!-- - `albumentations` — image augmentation -->
- `pandas` / `numpy` — data processing
- `scikit-learn` — preprocessing and TF-IDF
- `sqlite3` — built into Python, no install needed