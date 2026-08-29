# scripts/evaluate.py
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import torch
from tqdm import tqdm

from config import TUMOR2IDX

REPORTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'reports', 'figures')
os.makedirs(REPORTS_DIR, exist_ok=True)


def classification_eval(model, val_loader, device):
    """Print classification report and save confusion matrix."""
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, t_lbl, _, _, fv in tqdm(val_loader, desc="  Eval"):
            imgs, fv = imgs.to(device), fv.to(device)
            out_t, _, _ = model(imgs, fv)
            all_preds.extend(out_t.argmax(1).cpu().numpy())
            all_labels.extend(t_lbl.numpy())

    idx2tumor = {v: k for k, v in TUMOR2IDX.items()}
    names = [idx2tumor[i] for i in range(len(TUMOR2IDX))]
    print("\n[EVAL] Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=names, zero_division=0))

    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=names, yticklabels=names, ax=ax)
    ax.set_title('Confusion Matrix — Tumor Type')
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    path = os.path.join(REPORTS_DIR, 'confusion_matrix_final.png')
    fig.savefig(path, dpi=150)
    plt.close()
    print(f"[EVAL] Saved {path}")
    return all_labels, all_preds


def plot_roc_curve(model, val_loader, device, num_classes=10):
    """Multi-class ROC curve plot."""
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for imgs, t_lbl, _, _, fv in tqdm(val_loader, desc="  ROC"):
            imgs, fv = imgs.to(device), fv.to(device)
            out_t, _, _ = model(imgs, fv)
            all_probs.extend(torch.softmax(out_t, dim=1).cpu().numpy())
            all_labels.extend(t_lbl.numpy())

    y_true = label_binarize(all_labels, classes=list(range(num_classes)))
    y_pred = np.array(all_probs)
    idx2tumor = {v: k for k, v in TUMOR2IDX.items()}

    plt.figure(figsize=(12, 10))
    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(y_true[:, i], y_pred[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{idx2tumor[i]} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0, 1]); plt.ylim([0, 1.05])
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title('Multi-class ROC Curve')
    plt.legend(loc="lower right", fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(REPORTS_DIR, 'roc_curve.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[EVAL] Saved {path}")


def plot_training_curves(train_losses, val_losses, train_accs, val_accs):
    """Save training history plots."""
    epochs = range(1, len(train_losses) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epochs, train_losses, label='Train Loss', color='steelblue', linewidth=2)
    axes[0].plot(epochs, val_losses,   label='Val Loss',   color='coral',     linewidth=2)
    axes[0].set_title('Loss per Epoch'); axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)
    best_ep = val_losses.index(min(val_losses)) + 1
    axes[0].axvline(x=best_ep, color='gray', linestyle='--', alpha=0.7, label=f'Best epoch ({best_ep})')
    axes[0].legend()

    axes[1].plot(epochs, train_accs, label='Train Acc', color='steelblue', linewidth=2)
    axes[1].plot(epochs, val_accs,   label='Val Acc',   color='coral',     linewidth=2)
    axes[1].set_title('Tumor Accuracy per Epoch'); axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy (%)')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)
    best_acc_ep = val_accs.index(max(val_accs)) + 1
    axes[1].axvline(x=best_acc_ep, color='gray', linestyle='--', alpha=0.7, label=f'Best epoch ({best_acc_ep})')
    axes[1].legend()

    plt.tight_layout()
    path = os.path.join(REPORTS_DIR, 'training_curves.png')
    fig.savefig(path, dpi=150)
    plt.close()
    print(f"[PLOT] Saved {path}")
    print(f"[PLOT] Best val loss: {min(val_losses):.4f} @ epoch {best_ep}")
    print(f"[PLOT] Best val acc:  {max(val_accs):.2f}% @ epoch {best_acc_ep}")
