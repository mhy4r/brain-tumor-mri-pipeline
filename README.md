# Brain Tumor MRI Classification & MLOps Pipeline

An end-to-end Machine Learning & MLOps system for predicting **brain tumor types**, **MRI weightings**, and **lesion locations** from MRI scans. 

The project spans three comprehensive phases: **Database Design & Schema Modeling (Phase 1)**, **Data Preprocessing & Feature Engineering Pipeline (Phase 2)**, and **Multi-Task PyTorch Deep Learning with MLflow Experiment Tracking (Phase 3)**.

---

## 🏗️ Project Architecture & Pipeline Flow

```
[ Phase 1: Database Design ] ➔ Relational SQLite Database Schema (5 normalized tables)
                                      │
[ Phase 2: ETL Pipeline ]    ➔ DATA.json ➔ SQLite Ingestion ➔ Cleaning ➔ Feature Engineering
                                                                                      │
[ Phase 3: PyTorch & MLOps ] ➔ Multi-Task Neural Network ➔ MLflow Tracking ➔ Evaluation & CI
```

---

## 📊 Dataset & Setup

This project utilizes the [Brain Tumor MRI Images — 30 Classes](https://www.kaggle.com/datasets/fernando2rad/brain-tumor-mri-images-30-classes) dataset (distributed under Apache License 2.0).

- **11,300 MRI images** at 512×512 resolution.
- **10 Tumor Classes**: Astrocytoma, Ependymoma, Glioma, Hemangiopericytoma, Meningioma, Neurocytoma, Normal, Oligodendroglioma, Other, Schwannoma.
- **3 MRI Weightings**: T1, T1C+, T2.
- **Spatial Metadata**: Lesion center coordinates $(x, y)$ per image.

*Note: The raw dataset is omitted from git due to size.*

### Dataset Download Instructions
```bash
pip install kaggle
# Ensure kaggle.json is in ~/.kaggle/
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

---

## 📂 Repository Structure

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
│   ├── import_to_database.py    # ETL: DATA.json ➔ SQLite
│   ├── load_data.py             # SQLite DB reader ➔ Pandas DataFrame
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

---

## 🚀 Phase 1: Database Design & Architecture

Phase 1 models the complex MRI metadata into a 5-table normalized SQLite relational schema:

1. `images`: Primary image metadata (file path, image dimensions, creation date).
2. `tumor_types`: Mapping of tumor class names (0–9).
3. `tumor_grades`: Tumor severity/grade mappings.
4. `locations`: Anatomical brain region classifications.
5. `image_locations`: Spatial bounding box coordinates $(x, y)$ and lesion offsets.

---

## ⚙️ Phase 2: Data & Feature Engineering Pipeline

The ETL pipeline transforms raw JSON metadata into structured SQLite tables and engineered feature sets.

### Running Phase 2 Pipeline
```bash
python pipeline.py
```

### Engineered Features Summary
| Feature | Description |
|---------|-------------|
| `tumor_idx` | Ordinal encoding of tumor type (0–9) |
| `weight_idx` | Ordinal encoding of MRI weighting (0–2) |
| `point_x_norm` / `point_y_norm` | Lesion $X, Y$ coordinates normalized to $[0, 1]$ |
| `lesion_offset_norm` | Euclidean distance of lesion from image center (normalized) |
| `is_central` | Binary indicator ($1$ if lesion is within 20% of image center) |
| `tfidf_*` | Top 20 TF-IDF terms extracted from textual MRI descriptions |

---

## 🧠 Phase 3: Deep Learning Architecture & MLOps

Phase 3 implements a **Multi-Task Convolutional Neural Network (`CustomBrainNet`)** with three simultaneous output heads:

1. **Tumor Classification Head**: 10-class Softmax classification (CrossEntropy loss).
2. **Weighting Classification Head**: 3-class Softmax classification (T1 / T1C+ / T2).
3. **Heatmap Localization Head**: 2D Gaussian spatial heatmap regression supervising lesion center detection.

### Running Model Training with MLflow
```bash
python run_training_pipeline.py
```

### MLflow Experiment Tracking UI
Track loss curves, accuracy, per-class F1-scores, and saved model artifacts:
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
Open `http://localhost:5000` in your browser.

---

## 🔄 CI/CD Pipeline (GitHub Actions)

GitHub Actions runs automatically on every push or pull request to `main`.

The workflow:
1. Sets up **Python 3.12** environment.
2. Installs dependencies from `requirements.txt`.
3. Creates a **minimal synthetic SQLite database** to verify ETL functionality without full dataset download.
4. Runs `pipeline.py` to test data processing modules.
5. Executes a **smoke test** on `model.py` and `dataset.py` using synthetic tensor passes.

---

## 🛠️ Installation & Dependencies

```bash
pip install -r requirements.txt
```

Key dependencies:
- `torch` & `torchvision`: Multi-task neural network implementation
- `albumentations`: Image augmentations (affine, rotations, brightness)
- `mlflow`: Experiment tracking & model registry
- `pandas`, `numpy`, `scikit-learn`: Feature engineering & evaluation
- `sqlite3`: Native relational database engine
