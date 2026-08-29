# run_pipeline.py
# End-to-end pipeline: train → evaluate → predict → save predictions.
import os, sys, argparse

# Add p2/p3 scripts to path
P3_SCRIPTS = os.path.join(os.path.dirname(__file__), 'scripts')
P2_SCRIPTS = os.path.join(os.path.dirname(__file__), '..', 'p2', 'scripts')
for p in [P3_SCRIPTS, P2_SCRIPTS]:
    if p not in sys.path:
        sys.path.insert(0, p)

from config import CSV_PATH, DATASET_ROOT, MODEL_SAVE_PATH, TUMOR2IDX, WEIGHT2IDX, BATCH_SIZE, DEVICE
from dataset import prepare_data_list, get_stratified_loaders
from model import CustomBrainNet
from train_model import train
from make_predictions import load_best_model, batch_evaluate
from predictions import predict_and_save

import pandas as pd
import mlflow


def run_pipeline(mode='all', resume=False):
    """Orchestrate training, evaluation, and prediction."""
    print("=" * 60)
    print("[PIPELINE] Phase 3 — Multi-task Brain MRI Pipeline")
    print("=" * 60)

    # Load CSV and build feature columns
    df = pd.read_csv(CSV_PATH)
    LOC_COLS = [c for c in df.columns if c.startswith('loc_')]
    TFIDF_COLS = [c for c in df.columns if c.startswith('tfidf_')]
    FEATURE_COLS = ['desc_length'] + LOC_COLS + TFIDF_COLS
    feature_dim = len(FEATURE_COLS)

    # Build data
    data_list = prepare_data_list(df, DATASET_ROOT, FEATURE_COLS)
    train_loader, val_loader = get_stratified_loaders(data_list, batch_size=BATCH_SIZE)

    if mode in ('all', 'train'):
        print("\n--- TRAINING ---")
        with mlflow.start_run(run_name="brain_multi_task_v4"):
            model = CustomBrainNet(len(TUMOR2IDX), len(WEIGHT2IDX), feature_dim).to(DEVICE)
            train(model, train_loader, val_loader, DEVICE, MODEL_SAVE_PATH, resume_path=MODEL_SAVE_PATH if resume else None)
            mlflow.pytorch.log_model(model, "model", serialization_format="pickle")
            print("[MLFLOW] Training run logged.")

    if mode in ('all', 'eval'):
        print("\n--- EVALUATION ---")
        model = load_best_model(MODEL_SAVE_PATH, len(TUMOR2IDX), len(WEIGHT2IDX), feature_dim, DEVICE)
        batch_evaluate(model, val_loader, DEVICE)

    if mode in ('all', 'predict'):
        print("\n--- PREDICTION & DB SAVE ---")
        model = load_best_model(MODEL_SAVE_PATH, len(TUMOR2IDX), len(WEIGHT2IDX), feature_dim, DEVICE)
        df_preds = predict_and_save(model, val_loader, DEVICE)
        print(f"[PIPELINE] {len(df_preds)} predictions saved.")

    print("\n[PIPELINE] Done.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Phase 3 Pipeline')
    parser.add_argument('--mode', choices=['all', 'train', 'eval', 'predict'],
                        default='all', help='Pipeline stage(s) to run')
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from existing checkpoint')
    args = parser.parse_args()
    run_pipeline(mode=args.mode, resume=args.resume)
