# scripts/model.py
import torch
import torch.nn as nn

class CustomBrainNet(nn.Module):
    """
    Multi-task CNN: Conv backbone → spatial features + structured CSV features → fusion → 3 heads.
              tumor_head: 10-class   (3 linear layers, dropout)
             weight_head:  3-class   (2 linear layers)
            heatmap_head:  1-channel lesion heatmap   (decoder + sigmoid)
    """
    def __init__(self, num_tumor_classes=10, num_weight_classes=3, feature_dim=68):
        super().__init__()

        # Image backbone — 5 conv stages
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
        )  # (B, 512, 14, 14)

        # Heatmap decoder
        self.heatmap_head = nn.Sequential(
            nn.Conv2d(512, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 1, 1), nn.Sigmoid(),
        )

        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Feature MLP
        self.feature_mlp = nn.Sequential(
            nn.Linear(feature_dim, 128), nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(inplace=True),
        )

        fusion_dim = 512 + 64
        self.tumor_head = nn.Sequential(
            nn.Linear(fusion_dim, 256), nn.ReLU(inplace=True), nn.Dropout(0.4),
            nn.Linear(256, 128), nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(128, num_tumor_classes),
        )
        self.weight_head = nn.Sequential(
            nn.Linear(fusion_dim, 64), nn.ReLU(inplace=True),
            nn.Linear(64, num_weight_classes),
        )

    def forward(self, x, feat_vec):
        spatial = self.features(x)
        heatmap = self.heatmap_head(spatial)
        pooled = self.global_pool(spatial).view(x.size(0), -1)
        feat_e = self.feature_mlp(feat_vec)
        fused = torch.cat([pooled, feat_e], dim=1)
        return self.tumor_head(fused), self.weight_head(fused), heatmap
