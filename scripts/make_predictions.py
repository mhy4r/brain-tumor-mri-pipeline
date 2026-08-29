# scripts/make_predictions.py
import os, sys
import numpy as np
import cv2
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from config import IMG_SIZE, TUMOR2IDX, WEIGHT2IDX, DEVICE
from dataset import get_transforms
from model import CustomBrainNet


def load_best_model(path, num_tumor_classes, num_weight_classes, feature_dim, device):
    model = CustomBrainNet(num_tumor_classes, num_weight_classes, feature_dim).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    print(f"[LOAD] Model loaded from {path}")
    return model


def predict_single(model, image_path, feat_row, feature_cols, device):
    """Run inference on one image + CSV row."""
    idx2tumor = {v: k for k, v in TUMOR2IDX.items()}
    idx2weight = {v: k for k, v in WEIGHT2IDX.items()}

    image = cv2.imread(image_path)
    if image is None:
        print(f"[PREDICT] Cannot read {image_path}"); return
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transform = get_transforms(train=False)
    augmented = transform(image=image, keypoints=[])
    img_tensor = augmented['image'].unsqueeze(0).to(device)

    features = np.array([feat_row[c] for c in feature_cols], dtype=np.float32)
    feat_tensor = torch.tensor(features).unsqueeze(0).to(device)

    with torch.no_grad():
        out_t, out_w, out_hm = model(img_tensor, feat_tensor)
        t_probs = torch.softmax(out_t, dim=1)[0]
        w_probs = torch.softmax(out_w, dim=1)[0]
        t_pred, w_pred = t_probs.argmax().item(), w_probs.argmax().item()

    print(f"\n[PREDICT] {os.path.basename(image_path)}")
    print(f"  Tumor  → {idx2tumor[t_pred]} ({t_probs[t_pred]*100:.1f}%)")
    print(f"  Weight → {idx2weight[w_pred]} ({w_probs[w_pred]*100:.1f}%)")
    return idx2tumor[t_pred], idx2weight[w_pred], out_hm[0, 0].cpu().numpy()


def batch_evaluate(model, val_loader, device):
    """Full classification metrics on validation set."""
    from evaluate import classification_eval, plot_roc_curve
    classification_eval(model, val_loader, device)
    plot_roc_curve(model, val_loader, device)
