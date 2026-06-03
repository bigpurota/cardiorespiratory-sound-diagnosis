"""Deep-learning model factory."""
from src import config

import torch.nn as nn
import timm

from src.datasets import _to_effnet_image

__all__ = ["SmallCNN", "build_efficientnet_b0", "count_params", "_to_effnet_image"]


class SmallCNN(nn.Module):
    """From-scratch 4-conv-block CNN: ``(B, 1, 64, 128)`` ->"""

    def __init__(self, n_classes, p=0.3, widths=(16, 32, 64, 128)):
        super().__init__()
        if p < 0.3:
            raise ValueError(f"SmallCNN dropout p must be >= 0.3; got {p}")
        widths = tuple(widths)
        if len(widths) != 4:
            raise ValueError(f"SmallCNN widths must be a 4-tuple; got {widths}")

        def block(ci, co):
            return nn.Sequential(
                nn.Conv2d(ci, co, kernel_size=3, padding=1),
                nn.BatchNorm2d(co),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        w0, w1, w2, w3 = widths
        self.features = nn.Sequential(
            block(1, w0), block(w0, w1), block(w1, w2), block(w2, w3)
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p),
            nn.Linear(w3, n_classes),
        )

    def forward(self, x):
        return self.head(self.pool(self.features(x)))


def build_efficientnet_b0(n_classes, freeze_backbone=False):
    """Build the timm EfficientNet-B0 transfer model."""
    m = timm.create_model(
        "efficientnet_b0", pretrained=True, in_chans=3, num_classes=n_classes
    )
    if freeze_backbone:
        for p_ in m.parameters():
            p_.requires_grad = False
        for p_ in m.get_classifier().parameters():
            p_.requires_grad = True
    return m


def count_params(model):
    """Total parameter count of ``model``."""
    return int(sum(p.numel() for p in model.parameters()))
