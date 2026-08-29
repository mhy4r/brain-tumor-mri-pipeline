# scripts/train_model.py
import os, sys
import torch
import torch.nn as nn
from tqdm import tqdm
import mlflow

sys.path.insert(0, os.path.dirname(__file__))
from config import TUMOR2IDX, WEIGHT2IDX, DEVICE
from dataset import get_stratified_loaders
from model import CustomBrainNet
from evaluate import plot_training_curves


class BrainTrainer:
    def __init__(self, model, train_loader, val_loader, optimizer, scheduler, device):
        self.model, self.train_loader, self.val_loader = model, train_loader, val_loader
        self.optimizer, self.scheduler, self.device = optimizer, scheduler, device
        self.criterion_tumor = nn.CrossEntropyLoss()
        self.criterion_weight = nn.CrossEntropyLoss()
        self.criterion_heatmap = nn.MSELoss()

    def train_epoch(self):
        self.model.train()
        total_loss = correct = total = 0
        pbar = tqdm(self.train_loader, desc="  Train", leave=False)
        for imgs, t_lbl, w_lbl, hm, fv in pbar:
            imgs, t_lbl, w_lbl, hm, fv = [x.to(self.device) for x in (imgs, t_lbl, w_lbl, hm, fv)]
            self.optimizer.zero_grad()
            out_t, out_w, out_hm = self.model(imgs, fv)
            loss = (self.criterion_tumor(out_t, t_lbl) + 0.5 * self.criterion_weight(out_w, w_lbl)
                    + 5.0 * self.criterion_heatmap(out_hm, hm))
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            correct += (out_t.argmax(1) == t_lbl).sum().item()
            total += t_lbl.size(0)
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'acc': f"{100*correct/total:.1f}%"})
        return total_loss / len(self.train_loader), 100 * correct / total

    def validate(self):
        self.model.eval()
        val_loss = correct = total = 0
        with torch.no_grad():
            for imgs, t_lbl, w_lbl, hm, fv in self.val_loader:
                imgs, t_lbl, w_lbl, hm, fv = [x.to(self.device) for x in (imgs, t_lbl, w_lbl, hm, fv)]
                out_t, out_w, out_hm = self.model(imgs, fv)
                loss = (self.criterion_tumor(out_t, t_lbl) + 0.5 * self.criterion_weight(out_w, w_lbl)
                        + 5.0 * self.criterion_heatmap(out_hm, hm))
                val_loss += loss.item()
                correct += (out_t.argmax(1) == t_lbl).sum().item()
                total += t_lbl.size(0)
        return val_loss / len(self.val_loader), 100 * correct / total


class EarlyStopping:
    def __init__(self, patience=7, path='best_model.pth'):
        self.patience, self.path = patience, path
        self.counter, self.best_loss, self.early_stop = 0, None, False

    def __call__(self, val_loss, model):
        if self.best_loss is None or val_loss < self.best_loss:
            self.best_loss = val_loss
            torch.save(model.state_dict(), self.path)
            mlflow.log_artifact(self.path)
            print(f"  [CKPT] Val loss improved → {val_loss:.4f} — saved.")
            self.counter = 0
        else:
            self.counter += 1
            print(f"  [CKPT] No improvement {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True


def train(model, train_loader, val_loader, device, model_path, epochs=60, resume_path=None):
    if resume_path and os.path.exists(resume_path):
        model.load_state_dict(torch.load(resume_path, map_location=device))
        print(f"[TRAIN] Resumed from {resume_path}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    early_stopper = EarlyStopping(patience=7, path=model_path)

    # MLflow: log params
    mlflow.log_param("epochs", epochs)
    mlflow.log_param("batch_size", train_loader.batch_size)
    mlflow.log_param("learning_rate", 1e-4)
    mlflow.log_param("weight_decay", 1e-4)
    mlflow.log_param("optimizer", "AdamW")
    mlflow.log_param("scheduler", "CosineAnnealingLR")
    mlflow.log_param("image_size", 224)
    mlflow.log_param("tumor_loss_weight", 1.0)
    mlflow.log_param("weight_loss_weight", 0.5)
    mlflow.log_param("heatmap_loss_weight_train", 5.0)
    mlflow.log_param("early_stop_patience", 7)

    trainer = BrainTrainer(model, train_loader, val_loader, optimizer, scheduler, device)

    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        t_loss, t_acc = trainer.train_epoch()
        v_loss, v_acc = trainer.validate()
        scheduler.step()
        early_stopper(v_loss, model)
        train_losses.append(t_loss); val_losses.append(v_loss)
        train_accs.append(t_acc);   val_accs.append(v_acc)

        # MLflow: log metrics per epoch
        mlflow.log_metric("train_loss", t_loss, step=epoch)
        mlflow.log_metric("val_loss", v_loss, step=epoch)
        mlflow.log_metric("train_acc", t_acc, step=epoch)
        mlflow.log_metric("val_acc", v_acc, step=epoch)

        print(f"  Train — Loss: {t_loss:.4f} | Acc: {t_acc:.2f}%")
        print(f"  Val   — Loss: {v_loss:.4f} | Acc: {v_acc:.2f}%")
        if early_stopper.early_stop:
            print(f"[TRAIN] Early stopping at epoch {epoch}.")
            break

    # MLflow: log best metrics
    mlflow.log_metric("best_val_loss", min(val_losses))
    mlflow.log_metric("best_val_acc", max(val_accs))

    plot_training_curves(train_losses, val_losses, train_accs, val_accs)
    mlflow.log_artifact(os.path.join(os.path.dirname(__file__), '..', 'reports', 'figures'))


if __name__ == '__main__':
    from config import CSV_PATH, DATASET_ROOT, MODEL_SAVE_PATH, TUMOR2IDX, WEIGHT2IDX, BATCH_SIZE
    import pandas as pd

    df_meta = pd.read_csv(CSV_PATH)

    from dataset import prepare_data_list
    LOC_COLS = [c for c in df_meta.columns if c.startswith('loc_')]
    TFIDF_COLS = [c for c in df_meta.columns if c.startswith('tfidf_')]
    FEATURE_COLS = ['desc_length'] + LOC_COLS + TFIDF_COLS
    data_list = prepare_data_list(df_meta, DATASET_ROOT, FEATURE_COLS)
    train_loader, val_loader = get_stratified_loaders(data_list, batch_size=BATCH_SIZE)
    feature_dim = len(FEATURE_COLS)
    model = CustomBrainNet(len(TUMOR2IDX), len(WEIGHT2IDX), feature_dim).to(DEVICE)

    with mlflow.start_run(run_name="brain_multi_task_v4"):
        train(model, train_loader, val_loader, DEVICE, MODEL_SAVE_PATH)
        mlflow.pytorch.log_model(model, "model", serialization_format="pickle")
        print(f"[MLFLOW] Run logged. View with: mlflow ui")
