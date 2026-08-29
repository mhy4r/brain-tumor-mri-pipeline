# Brain Tumor MRI Classification & MLOps Pipeline

An end to end machine learning and MLOps pipeline for predicting brain tumor types, MRI weightings, and lesion locations from MRI scans.

The project is structured in three phases: database design and schema modeling (Phase 1), data preprocessing and feature engineering (Phase 2), and multi-task PyTorch deep learning with MLflow tracking (Phase 3).

## Project architecture

```
[ Phase 1: Database Design ] -> Relational SQLite Database Schema (5 normalized tables)
                                      |
[ Phase 2: ETL Pipeline ]    -> DATA.json -> SQLite Ingestion -> Cleaning -> Feature Engineering
                                                                                      |
[ Phase 3: PyTorch & MLOps ] -> Multi-Task Neural Network -> MLflow Tracking -> Evaluation & CI
```

## Dataset and setup

This project uses the [Brain Tumor MRI Images — 30 Classes](https://www.kaggle.com/datasets/fernando2rad/brain-tumor-mri-images-30-classes) dataset (distributed under Apache License 2.0).

- **11,300 MRI images** at 512x512 resolution.
- **10 tumor classes**: Astrocytoma, Ependymoma, Glioma, Hemangiopericytoma, Meningioma, Neurocytoma, Normal, Oligodendroglioma, Other, Schwannoma.
- **3 MRI weightings**: T1, T1C+, T2.
- **Spatial metadata**: Lesion center coordinates (x, y) per image.

Note: The raw dataset is omitted from git tracking due to file size.

### Dataset download

```bash
pip install kaggle
# Ensure kaggle.json is placed in ~/.kaggle/
kaggle datasets download -d fernando2rad/brain-tumor-mri-images-30-classes --unzip
```

Expected project file structure:

```
project/
├── dataset/             # Raw image folders
├── DATA.json            # Raw metadata JSON
└── database/
    └── brain_tumor_db.sqlite
```

## Repository structure

```
brain-tumor-mri-pipeline/
├── .github/
│   └── workflows/
│       └── ci.yml               # GitHub Actions CI/CD Pipeline (ETL + Model Smoke Test)
├── docs/                        # Phase 1 Schema specifications & architecture docs
├── database/
│   └── brain_tumor_db.sqlite    # SQLite database (generated during ETL)
├── scripts/
│   ├── database_connection.py   # DB helper module
│   ├── import_to_database.py    # ETL: DATA.json -> SQLite
│   ├── load_data.py             # SQLite DB reader -> Pandas DataFrame
│   ├── preprocess.py            # Data cleaning & normalization
│   ├── feature_engineering.py   # Spatial & NLP TF-IDF feature extraction
│   ├── config.py                # Hyperparameters, paths, & class mappings
│   ├── dataset.py               # Custom PyTorch Dataset & Albumentations
│   ├── model.py                 # Multi-task CNN architecture
│   ├── train_model.py           # PyTorch training loop + MLflow tracking
│   ├── evaluate.py              # Per-class evaluation & metric visualizer
│   └── predictions.py           # Inference runner
├── pipeline.py                  # Phase 2 ETL Master Entry Point
├── run_training_pipeline.py     # Phase 3 Model Training Entry Point
├── notebooks/                   # Research & exploratory notebooks
├── reports/                     # Pre-rendered evaluation plots & confusion matrices
└── requirements.txt
```

## Phase 1: Database design and architecture

Phase 1 models the MRI metadata into a 5-table normalized SQLite relational schema:

1. `images`: Primary image metadata (file path, image dimensions, creation date).
2. `tumor_types`: Mapping of tumor class names (0 to 9).
3. `tumor_grades`: Tumor severity and grade mappings.
4. `locations`: Anatomical brain region classifications.
5. `image_locations`: Spatial bounding box coordinates (x, y) and lesion offsets.

## Phase 2: Data and feature engineering pipeline

The ETL pipeline transforms raw JSON metadata into structured SQLite tables and engineered feature sets.

### Running Phase 2 pipeline

```bash
python pipeline.py
```

### Feature engineering summary

| Feature | Description |
|---------|-------------|
| `tumor_idx` | Ordinal encoding of tumor type (0 to 9) |
| `weight_idx` | Ordinal encoding of MRI weighting (0 to 2) |
| `point_x_norm` / `point_y_norm` | Lesion X, Y coordinates normalized to [0, 1] |
| `lesion_offset_norm` | Euclidean distance of lesion from image center (normalized) |
| `is_central` | Binary indicator (1 if lesion is within 20% of image center) |
| `tfidf_*` | Top 20 TF-IDF terms extracted from textual MRI descriptions |

## Phase 3: Deep learning architecture and MLOps

Phase 3 implements a multi-task convolutional neural network (`CustomBrainNet`) with three output heads:

1. **Tumor classification head**: 10-class Softmax classification using cross-entropy loss.
2. **Weighting classification head**: 3-class Softmax classification (T1, T1C+, T2).
3. **Heatmap localization head**: 2D Gaussian spatial heatmap regression supervising lesion center detection.

### Running model training with MLflow

```bash
python run_training_pipeline.py
```

### MLflow experiment tracking UI

Track loss curves, accuracy, per-class F1 scores, and saved model artifacts:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Open `http://localhost:5000` in your browser.

## CI/CD pipeline (GitHub Actions)

GitHub Actions runs automatically on every push or pull request to `main`.

The workflow steps:
1. Sets up the Python 3.12 environment.
2. Installs dependencies from `requirements.txt`.
3. Creates a minimal synthetic SQLite database to verify ETL functionality.
4. Runs `pipeline.py` to test data processing modules.
5. Executes a smoke test on `model.py` and `dataset.py` using synthetic tensor passes.

## Installation and dependencies

```bash
pip install -r requirements.txt
```

Key dependencies:
- `torch` & `torchvision`: Multi-task neural network implementation
- `albumentations`: Image augmentations (affine, rotations, brightness)
- `mlflow`: Experiment tracking & model registry
- `pandas`, `numpy`, `scikit-learn`: Feature engineering & evaluation
- `sqlite3`: Native relational database engine
