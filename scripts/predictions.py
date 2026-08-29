# scripts/predictions.py
import os, sys, sqlite3
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(__file__))
from config import CSV_PATH, DB_PATH, MODEL_SAVE_PATH, DATASET_ROOT, TUMOR2IDX, WEIGHT2IDX, DEVICE
from dataset import prepare_data_list, get_stratified_loaders
from make_predictions import load_best_model


def predict_and_save(model, val_loader, device, output_table='predictions'):
    """Run inference on val loader, save results to database."""
    model.eval()
    results = []
    idx2tumor = {v: k for k, v in TUMOR2IDX.items()}
    idx2weight = {v: k for k, v in WEIGHT2IDX.items()}

    with torch.no_grad():
        for batch_idx, (imgs, t_lbl, w_lbl, _, fv) in enumerate(val_loader):
            imgs, fv = imgs.to(device), fv.to(device)
            out_t, out_w, _ = model(imgs, fv)
            t_probs = torch.softmax(out_t, dim=1)
            w_probs = torch.softmax(out_w, dim=1)
            t_preds = t_probs.argmax(1).cpu().numpy()
            w_preds = w_probs.argmax(1).cpu().numpy()
            t_confs = t_probs.max(1).values.cpu().numpy()
            w_confs = w_probs.max(1).values.cpu().numpy()

            for i in range(len(t_preds)):
                results.append({
                    'batch_idx': batch_idx,
                    'sample_idx': i,
                    'true_tumor': idx2tumor[t_lbl[i].item()],
                    'pred_tumor': idx2tumor[t_preds[i]],
                    'tumor_confidence': float(t_confs[i]),
                    'true_weight': idx2weight[w_lbl[i].item()],
                    'pred_weight': idx2weight[w_preds[i]],
                    'weight_confidence': float(w_confs[i]),
                })

    # Save to SQLite
    con = sqlite3.connect(DB_PATH)
    df_out = pd.DataFrame(results)
    df_out.to_sql(output_table, con, if_exists='replace', index=False)
    con.close()
    print(f"[PREDICTIONS] Saved {len(results)} predictions to DB table '{output_table}'")
    return df_out


if __name__ == '__main__':
    df = pd.read_csv(CSV_PATH)
    LOC_COLS = [c for c in df.columns if c.startswith('loc_')]
    TFIDF_COLS = [c for c in df.columns if c.startswith('tfidf_')]
    FEATURE_COLS = ['desc_length'] + LOC_COLS + TFIDF_COLS
    data_list = prepare_data_list(df, DATASET_ROOT, FEATURE_COLS)
    _, val_loader = get_stratified_loaders(data_list)

    model = load_best_model(MODEL_SAVE_PATH, len(TUMOR2IDX), len(WEIGHT2IDX),
                            len(FEATURE_COLS), DEVICE)
    predict_and_save(model, val_loader, DEVICE)
