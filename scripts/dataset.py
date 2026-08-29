# scripts/dataset.py
import os, re
import numpy as np
import pandas as pd
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from config import IMG_SIZE, BATCH_SIZE, TUMOR2IDX, WEIGHT2IDX


def create_heatmap(shape, center, sigma=5):
    """Gaussian heatmap centered on lesion point. Zeros for Normal scans."""
    heatmap = np.zeros(shape, dtype=np.float32)
    if center is None or center[0] == -1:
        return heatmap
    x0, y0 = center
    y, x = np.ogrid[:shape[0], :shape[1]]
    heatmap = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    return heatmap


def get_transforms(train=True):
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    if train:
        return A.Compose([
            A.Resize(IMG_SIZE, IMG_SIZE),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
    return A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))


class BrainMRIDataset(Dataset):
    def __init__(self, data_list, feature_cols, transform=None):
        self.data = data_list
        self.feature_cols = feature_cols
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = cv2.imread(item['image_path'])
        if image is None:
            return self.__getitem__((idx + 1) % len(self.data))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        kp = item['keypoint']

        if self.transform:
            if kp[0] != -1:
                augmented = self.transform(image=image, keypoints=[kp])
                image = augmented['image']
                kp = augmented['keypoints'][0] if augmented['keypoints'] else (-1, -1)
            else:
                augmented = self.transform(image=image, keypoints=[])
                image = augmented['image']

        hmap = create_heatmap((IMG_SIZE, IMG_SIZE), kp)
        hmap = torch.tensor(hmap, dtype=torch.float32).unsqueeze(0)
        feat_vec = torch.tensor(item['features'], dtype=torch.float32)
        t_label = torch.tensor(TUMOR2IDX[item['tumor_label']], dtype=torch.long)
        w_label = torch.tensor(WEIGHT2IDX[item['weight_label']], dtype=torch.long)
        return image, t_label, w_label, hmap, feat_vec


def prepare_data_list(df, dataset_root, feature_cols):
    """Build list of dicts from processed_features.csv."""
    data_list = []
    skipped = 0
    for _, row in df.iterrows():
        clean_path = str(row['file_name']).replace('\\', os.sep).replace('/', os.sep)
        full_img_path = os.path.join(dataset_root, clean_path)
        if not os.path.exists(full_img_path):
            skipped += 1
            continue
        raw_class = str(row['class'])
        weight_label = 'Unknown'
        for w in ['T1C+', 'T2', 'T1']:
            if raw_class.endswith(w):
                weight_label = w
                break
        tumor_label = str(row['tumor_type']).strip()
        kp = (-1, -1) if tumor_label == 'Normal' else (float(row['point_x']), float(row['point_y']))
        features = row[feature_cols].values.astype(np.float32)
        data_list.append({
            'image_path': full_img_path,
            'tumor_label': tumor_label,
            'weight_label': weight_label,
            'keypoint': kp,
            'features': features,
        })
    print(f"[DATASET] Loaded {len(data_list)} samples, skipped {skipped}")
    return data_list


def get_stratified_loaders(data_list, batch_size=BATCH_SIZE, test_size=0.2, seed=42):
    labels = [f"{d['tumor_label']}_{d['weight_label']}" for d in data_list]
    train_data, val_data = train_test_split(data_list, test_size=test_size,
                                            random_state=seed, stratify=labels)
    train_ds = BrainMRIDataset(train_data, None, transform=get_transforms(train=True))
    val_ds   = BrainMRIDataset(val_data,   None, transform=get_transforms(train=False))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    print(f"[DATASET] Train {len(train_data)} | Val {len(val_data)}")
    return train_loader, val_loader
