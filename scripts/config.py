# scripts/config.py
import os, sys
import torch

P2_SCRIPTS = os.path.join(os.path.dirname(__file__), '..', '..', 'p2', 'scripts')
if P2_SCRIPTS not in sys.path:
    sys.path.insert(0, P2_SCRIPTS)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
P2_ROOT      = os.path.join(PROJECT_ROOT, '..', 'p2')
DATASET_ROOT = os.path.join(PROJECT_ROOT, '..', 'dataset')
CSV_PATH     = os.path.join(P2_ROOT, 'processed_features.csv')
DB_PATH      = os.path.join(P2_ROOT, 'database', 'brain_tumor_db.sqlite')

IMG_SIZE     = 224
BATCH_SIZE   = 32
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TUMOR2IDX = {
    'Normal': 0, 'Astrocytoma': 1, 'Ependymoma': 2, 'Glioma': 3,
    'Hemangiopericytoma': 4, 'Meningioma': 5, 'Neurocytoma': 6,
    'Oligodendroglioma': 7, 'Schwannoma': 8, 'Other': 9,
}
WEIGHT2IDX = {'T1': 0, 'T2': 1, 'T1C+': 2}

MODEL_SAVE_PATH = os.path.join(PROJECT_ROOT, 'models', 'best_model_v4.pth')
